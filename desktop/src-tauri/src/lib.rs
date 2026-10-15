// Intro Coach desktop shell: the UI and all analysis run in the webview; Rust only downloads and serves the
// 290 MB speech model from the app's data folder, and provides an HTTP client without browser CORS limits.
use futures_util::StreamExt;
use serde::Serialize;
use std::path::PathBuf;
use tauri::{ipc::Channel, ipc::Response, AppHandle, Manager};
use tokio::io::AsyncWriteExt;

#[derive(Clone, Serialize)]
struct Progress {
    loaded: u64,
    total: u64,
}

fn model_dir(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = app.path().app_data_dir().map_err(|e| e.to_string())?.join("models").join("fluid-speech-mini-onnx");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

/// Download any model files that aren't on disk yet, streaming progress to the UI.
#[tauri::command]
async fn ensure_model(app: AppHandle, repo: String, files: Vec<String>, on_progress: Channel<Progress>) -> Result<(), String> {
    let dir = model_dir(&app)?;
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
        tokio::fs::rename(&tmp, dir.join(f)).await.map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// One model file as raw bytes (sent to the webview without JSON encoding).
#[tauri::command]
async fn read_model_file(app: AppHandle, name: String) -> Result<Response, String> {
    if name.contains('/') || name.contains('\\') || name.contains("..") {
        return Err("bad file name".into());
    }
    let bytes = tokio::fs::read(model_dir(&app)?.join(&name)).await.map_err(|e| e.to_string())?;
    Ok(Response::new(bytes))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_http::init())
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(tauri_plugin_log::Builder::default().level(log::LevelFilter::Info).build())?;
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![ensure_model, read_model_file])
        .run(tauri::generate_context!())
        .expect("error while running Intro Coach");
}
