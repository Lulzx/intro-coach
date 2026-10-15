// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "coreml-asr",
    platforms: [.macOS(.v12)],
    targets: [.executableTarget(name: "coreml-asr", path: "Sources/coreml-asr")]
)
