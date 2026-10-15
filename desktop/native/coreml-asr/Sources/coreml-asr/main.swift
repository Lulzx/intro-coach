// Fluid Speech Mini (FluidVoice, CC BY 4.0) on Core ML: the Neural Engine on Apple silicon.
//
//   coreml-asr <model folder>
//
// Loads the four models once, prints {"ready":true} and then serves requests: each stdin line is the path of a
// file of 16 kHz mono float32 samples; each reply is one JSON line {"pieces":[[text,start,end],...],"ms":n}
// (subword pieces with times in seconds, the same shape as the ONNX engine) or {"error":"..."}.
import CoreML
import Foundation

let windowSamples = 240_000 // the models take fixed 15 s windows
let rate = 16_000
let maxSymbolsPerFrame = 10

struct Failure: Error, CustomStringConvertible { let description: String }

func emit(_ obj: [String: Any]) {
    let data = try! JSONSerialization.data(withJSONObject: obj)
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write("\n".data(using: .utf8)!)
}

func array(_ shape: [Int], _ type: MLMultiArrayDataType) throws -> MLMultiArray {
    try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: type)
}

/// Float value of a multi-array element at a flat (contiguous, row-major) index, whatever its storage type.
func reader(_ a: MLMultiArray) -> (Int) -> Float {
    switch a.dataType {
    case .float32:
        let p = a.dataPointer.assumingMemoryBound(to: Float.self)
        return { p[$0] }
    case .double:
        let p = a.dataPointer.assumingMemoryBound(to: Double.self)
        return { Float(p[$0]) }
    case .int32:
        let p = a.dataPointer.assumingMemoryBound(to: Int32.self)
        return { Float(p[$0]) }
    default:
        #if arch(arm64)
        let p = a.dataPointer.assumingMemoryBound(to: Float16.self)
        return { Float(p[$0]) }
        #else
        return { a[$0].floatValue }
        #endif
    }
}

func int(_ a: MLMultiArray) -> Int { Int(reader(a)(0)) }

final class Engine {
    let pre: MLModel, enc: MLModel, dec: MLModel, joint: MLModel
    let vocab: [Int: String]
    let blank: Int

    init(folder: URL) throws {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all
        func load(_ name: String) throws -> MLModel {
            try MLModel(contentsOf: folder.appendingPathComponent("\(name).mlmodelc"), configuration: cfg)
        }
        pre = try load("Preprocessor")
        enc = try load("Encoder")
        dec = try load("Decoder")
        joint = try load("JointDecision")
        let config = try JSONSerialization.jsonObject(with: Data(contentsOf: folder.appendingPathComponent("config.json"))) as? [String: Any] ?? [:]
        blank = config["blank_id"] as? Int ?? 8192
        let raw = try JSONSerialization.jsonObject(with: Data(contentsOf: folder.appendingPathComponent("parakeet_vocab.json"))) as? [String: String] ?? [:]
        var v: [Int: String] = [:]
        for (k, s) in raw { if let i = Int(k) { v[i] = s } }
        vocab = v
    }

    private func decoderStep(_ token: Int, _ h: MLMultiArray, _ c: MLMultiArray) throws -> (MLMultiArray, MLMultiArray, MLMultiArray) {
        let targets = try array([1, 1], .int32); targets[0] = NSNumber(value: token)
        let length = try array([1], .int32); length[0] = 1
        let out = try dec.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "targets": targets, "target_length": length, "h_in": h, "c_in": c,
        ]))
        return (out.featureValue(for: "decoder")!.multiArrayValue!, out.featureValue(for: "h_out")!.multiArrayValue!,
                out.featureValue(for: "c_out")!.multiArrayValue!)
    }

    /// One window of at most 15 s: features, encoder, then greedy token-and-duration decoding.
    func window(_ samples: ArraySlice<Float>, offset: Double) throws -> [[Any]] {
        let audio = try array([1, windowSamples], .float32)
        let ap = audio.dataPointer.assumingMemoryBound(to: Float.self)
        ap.initialize(repeating: 0, count: windowSamples)
        for (i, x) in samples.enumerated() { ap[i] = x }
        let alen = try array([1], .int32); alen[0] = NSNumber(value: samples.count)
        let mel = try pre.prediction(from: MLDictionaryFeatureProvider(dictionary: ["audio_signal": audio, "audio_length": alen]))
        let encOut = try enc.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "mel": mel.featureValue(for: "mel")!.multiArrayValue!, "mel_length": mel.featureValue(for: "mel_length")!.multiArrayValue!,
        ]))
        let encoder = encOut.featureValue(for: "encoder")!.multiArrayValue! // [1, 1024, frames]
        let frames = encoder.shape[2].intValue, hidden = encoder.shape[1].intValue
        let valid = min(int(encOut.featureValue(for: "encoder_length")!.multiArrayValue!), frames)
        let frameSeconds = Double(windowSamples) / Double(rate) / Double(frames)
        let read = reader(encoder)
        let s1 = encoder.strides[1].intValue, s2 = encoder.strides[2].intValue

        let step = try array([1, hidden, 1], .float32)
        let sp = step.dataPointer.assumingMemoryBound(to: Float.self)
        var h = try array([2, 1, 640], .float32), c = try array([2, 1, 640], .float32)
        for a in [h, c] { a.dataPointer.assumingMemoryBound(to: Float.self).initialize(repeating: 0, count: 1280) }
        var decOut: MLMultiArray
        (decOut, h, c) = try decoderStep(blank, h, c)

        var pieces: [[Any]] = []
        var t = 0, symbolsHere = 0, lastFrame = -1
        while t < valid {
            if t != lastFrame {
                for k in 0..<hidden { sp[k] = read(k * s1 + t * s2) }
                lastFrame = t
            }
            let out = try joint.prediction(from: MLDictionaryFeatureProvider(dictionary: ["encoder_step": step, "decoder_step": decOut]))
            let token = int(out.featureValue(for: "token_id")!.multiArrayValue!)
            var dur = int(out.featureValue(for: "duration")!.multiArrayValue!)
            if token != blank {
                let start = offset + Double(t) * frameSeconds
                if var text = vocab[token], !text.hasPrefix("<") {
                    // this vocabulary marks a word start with a space; use the SentencePiece marker like the ONNX engine
                    if text.hasPrefix(" ") { text = "\u{2581}" + text.dropFirst() }
                    pieces.append([text, start, start + Double(max(dur, 1)) * frameSeconds])
                }
                (decOut, h, c) = try decoderStep(token, h, c)
                symbolsHere += 1
            }
            if token == blank && dur == 0 { dur = 1 }
            if dur == 0 && symbolsHere >= maxSymbolsPerFrame { dur = 1 }
            if dur > 0 { symbolsHere = 0 }
            t += dur
        }
        return pieces
    }

    /// Cut audio longer than one window at the quietest point a little before each 15 s boundary.
    func transcribe(_ wav: [Float]) throws -> [[Any]] {
        var pieces: [[Any]] = []
        var start = 0
        let hop = rate / 50
        while start < wav.count {
            var end = min(wav.count, start + windowSamples)
            if end < wav.count {
                // search the last 3 s of the window for the quietest 20 ms
                var best = end, bestE = Float.greatestFiniteMagnitude
                var i = end - 3 * rate
                while i + hop <= end {
                    var e: Float = 0
                    for j in i..<(i + hop) { e += wav[j] * wav[j] }
                    if e < bestE { bestE = e; best = i + hop / 2 }
                    i += hop
                }
                end = best
            }
            pieces += try window(wav[start..<end], offset: Double(start) / Double(rate))
            start = end
        }
        return pieces
    }
}

// ---------------------------------------------------------------- main

guard CommandLine.arguments.count == 2 else {
    emit(["error": "usage: coreml-asr <model folder>"]); exit(2)
}
guard #available(macOS 15, *) else {
    emit(["error": "Core ML speech model needs macOS 15 or later"]); exit(3)
}
let engine: Engine
do {
    let t0 = Date()
    engine = try Engine(folder: URL(fileURLWithPath: CommandLine.arguments[1]))
    emit(["ready": true, "load_ms": Int(Date().timeIntervalSince(t0) * 1000)])
} catch {
    emit(["error": "couldn't load the Core ML models: \(error)"]); exit(4)
}

while let line = readLine(strippingNewline: true) {
    let path = line.trimmingCharacters(in: .whitespaces)
    if path.isEmpty { continue }
    do {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let wav: [Float] = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        let t0 = Date()
        let pieces = try engine.transcribe(wav)
        emit(["pieces": pieces, "ms": Int(Date().timeIntervalSince(t0) * 1000)])
    } catch {
        emit(["error": "\(error)"])
    }
}
