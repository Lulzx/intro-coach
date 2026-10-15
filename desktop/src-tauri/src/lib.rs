// Intro Coach desktop shell: the UI and the analysis run in the webview. Rust downloads and serves the speech
// model from the app's data folder, provides an HTTP client without browser CORS limits, and on macOS runs the
// Core ML build of the model on the Neural Engine through a small Swift helper (native/coreml-asr).
use futures_util::StreamExt;
use serde::Serialize;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Mutex;
use tauri::{ipc::Channel, ipc::Request, ipc::Response, AppHandle, Manager, State};
use tokio::io::AsyncWriteExt;

#[derive(Clone, Serialize)]
struct Progress {
    loaded: u64,
    total: u64,
}

fn model_dir(app: &AppHandle, repo: &str) -> Result<PathBuf, String> {
    let name = repo.rsplit('/').next().unwrap_or(repo);
    let dir = app.path().app_data_dir().map_err(|e| e.to_string())?.join("models").join(name);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

/// Download any model files that aren't on disk yet, streaming progress to the UI.
#[tauri::command]
async fn ensure_model(app: AppHandle, repo: String, files: Vec<String>, on_progress: Channel<Progress>) -> Result<(), String> {
    let dir = model_dir(&app, &repo)?;
    let missing: Vec<String> = files.into_iter().filter(|f| !dir.join(f).exists()).collect();
    if missing.is_empty() {
        let _ = on_progress.send(Progress { loaded: 1, total: 1 });
        return Ok(());
    }
    let client = reqwest::Client::new();
    let url = |f: &str| format!("https://huggingface.co/{repo}/resolve/main/{f}");
    let mut total = 0u64;
    for f in &missing {
        let head = client.head(url(f)).send().await.map_err(|e| e.to_string())?;
        total += head.headers().get("content-length").and_then(|v| v.to_str().ok()?.parse().ok()).unwrap_or(0);
    }
    let mut loaded = 0u64;
    for f in &missing {
        let res = client.get(url(f)).send().await.map_err(|e| e.to_string())?;
        if !res.status().is_success() {
            return Err(format!("couldn't download {f}: {}", res.status()));
        }
        // write to a temp file and rename, so a cancelled download never leaves a half file behind
        let target = dir.join(f);
        if let Some(parent) = target.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| e.to_string())?;
        }
        let tmp = dir.join(format!("{f}.part"));
        let mut out = tokio::fs::File::create(&tmp).await.map_err(|e| e.to_string())?;
        let mut stream = res.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| e.to_string())?;
            out.write_all(&chunk).await.map_err(|e| e.to_string())?;
            loaded += chunk.len() as u64;
            let _ = on_progress.send(Progress { loaded, total });
        }
        out.flush().await.map_err(|e| e.to_string())?;
        tokio::fs::rename(&tmp, &target).await.map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// One model file as raw bytes (sent to the webview without JSON encoding).
#[tauri::command]
async fn read_model_file(app: AppHandle, repo: String, name: String) -> Result<Response, String> {
    if name.contains('/') || name.contains('\\') || name.contains("..") {
        return Err("bad file name".into());
    }
    let bytes = tokio::fs::read(model_dir(&app, &repo)?.join(&name)).await.map_err(|e| e.to_string())?;
    Ok(Response::new(bytes))
}

// ---------------------------------------------------------------- Core ML speech recognition (macOS)

struct Native {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

#[derive(Default)]
struct NativeAsr(Mutex<Option<Native>>);

fn read_reply(stdout: &mut BufReader<ChildStdout>) -> Result<serde_json::Value, String> {
    let mut line = String::new();
    if stdout.read_line(&mut line).map_err(|e| e.to_string())? == 0 {
        return Err("the Core ML helper stopped".into());
    }
    let v: serde_json::Value = serde_json::from_str(&line).map_err(|e| e.to_string())?;
    match v.get("error").and_then(|e| e.as_str()) {
        Some(err) => Err(err.to_string()),
        None => Ok(v),
    }
}

/// Start the Core ML helper (bundled next to the app's executable) with the downloaded model. Fails on Windows,
/// on macOS before 15, or if the models don't load; the app then uses the ONNX engine instead.
#[tauri::command]
async fn native_asr_start(app: AppHandle, state: State<'_, NativeAsr>, repo: String) -> Result<serde_json::Value, String> {
    if !cfg!(target_os = "macos") {
        return Err("Core ML is only available on macOS".into());
    }
    let mut guard = state.0.lock().map_err(|e| e.to_string())?;
    if let Some(native) = guard.as_mut() {
        if native.child.try_wait().map_err(|e| e.to_string())?.is_none() {
            return Ok(serde_json::json!({ "ready": true }));
        }
    }
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let helper = exe.parent().ok_or("no executable folder")?.join("coreml-asr");
    let mut child = Command::new(&helper)
        .arg(model_dir(&app, &repo)?)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("couldn't start {}: {e}", helper.display()))?;
    let stdin = child.stdin.take().ok_or("no stdin")?;
    let mut stdout = BufReader::new(child.stdout.take().ok_or("no stdout")?);
    let ready = read_reply(&mut stdout)?;
    *guard = Some(Native { child, stdin, stdout });
    Ok(ready)
}

/// Transcribe 16 kHz mono float32 samples (sent as the raw request body) with the Core ML helper.
#[tauri::command]
async fn native_asr_transcribe(app: AppHandle, state: State<'_, NativeAsr>, request: Request<'_>) -> Result<serde_json::Value, String> {
    let tauri::ipc::InvokeBody::Raw(bytes) = request.body() else {
        return Err("expected raw audio bytes".into());
    };
    let path = app.path().app_cache_dir().map_err(|e| e.to_string())?.join("take.f32");
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(&path, bytes).map_err(|e| e.to_string())?;
    let mut guard = state.0.lock().map_err(|e| e.to_string())?;
    let native = guard.as_mut().ok_or("the Core ML helper isn't running")?;
    writeln!(native.stdin, "{}", path.display()).map_err(|e| e.to_string())?;
    native.stdin.flush().map_err(|e| e.to_string())?;
    let reply = read_reply(&mut native.stdout);
    let _ = std::fs::remove_file(&path);
    reply
}

// ---------------------------------------------------------------- self-test
// INTRO_COACH_SELFTEST=<audio file> makes the app analyse that file through the real pipeline, print a summary
// and quit. Used to check a build end to end without clicking through the window.

#[tauri::command]
fn selftest_file() -> Result<Response, String> {
    match std::env::var("INTRO_COACH_SELFTEST") {
        Ok(path) => std::fs::read(path).map(Response::new).map_err(|e| e.to_string()),
        Err(_) => Err("no self-test".into()),
    }
}

#[tauri::command]
fn selftest_done(app: AppHandle, summary: String) {
    println!("SELFTEST {summary}");
    app.exit(0);
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_http::init())
        .manage(NativeAsr::default())
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(tauri_plugin_log::Builder::default().level(log::LevelFilter::Info).build())?;
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![ensure_model, read_model_file, native_asr_start, native_asr_transcribe, selftest_file, selftest_done])
        .run(tauri::generate_context!())
        .expect("error while running Intro Coach");
}
