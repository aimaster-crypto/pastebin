import os
import subprocess
import json
import threading
import time
import re
import shutil
from typing import List

def ensure_maven_compiled(project_root: str):
    print("[DEBUG] Compiling Maven project...")
    try:
        result = subprocess.run([
            "./mvnw", "clean", "compile", "-q"
        ], cwd=project_root, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("[DEBUG] Maven compile successful")
            return True
        else:
            print(f"[WARNING] Maven compile failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"[WARNING] Could not compile Maven project: {e}")
        return False

def get_java_errors(java_file: str, project_root: str, jdtls_path: str, config_dir: str, workspace_dir: str) -> List[str]:
    errors = {}
    project_imported = False

    def send(proc, msg):
        body = json.dumps(msg)
        proc.stdin.write(f"Content-Length: {len(body)}\r\n\r\n".encode())
        proc.stdin.write(body.encode())
        proc.stdin.flush()

    def on_message(msg):
        nonlocal project_imported
        method = msg.get('method')

        if method == "textDocument/publishDiagnostics":
            uri = msg["params"]["uri"]
            errs = msg["params"]["diagnostics"]
            errors[uri] = errs
            print(f"[DEBUG] Diagnostics for {os.path.basename(uri)}: {len(errs)} issues")

        elif method == "language/status":
            status = msg.get("params", {}).get("message", "")
            print(f"[DEBUG] Status: {status}")
            if "Ready" in status or "Workspace projects imported" in status or "ServiceReady" in status:
                project_imported = True

    def reader(proc):
        buf = b""
        while True:
            try:
                byte = proc.stdout.read(1)
                if not byte:
                    break
                buf += byte
                if b"\r\n\r\n" in buf:
                    hdr, _, buf = buf.partition(b"\r\n\r\n")
                    match = re.search(br"Content-Length: (\d+)", hdr)
                    if match:
                        length = int(match.group(1))
                        body = proc.stdout.read(length)
                        try:
                            msg = json.loads(body.decode())
                            on_message(msg)
                        except json.JSONDecodeError:
                            pass
            except:
                break

    def stderr_reader(proc):
        try:
            for line in proc.stderr:
                print(f"[JDT MESSAGE] {line.decode().strip()}")
        except:
            pass

    # Step 1: Compile the Maven project
    ensure_maven_compiled(project_root)

    # Step 2: Clean the workspace (use /tmp to avoid overlap issue)
    if os.path.exists(workspace_dir):
        print(f"[DEBUG] Removing existing workspace: {workspace_dir}")
        shutil.rmtree(workspace_dir)

    # Step 3: Locate the launcher jar
    plugins_dir = os.path.join(jdtls_path, "plugins")
    launcher_jars = [f for f in os.listdir(plugins_dir) if "org.eclipse.equinox.launcher" in f and f.endswith(".jar")]
    if not launcher_jars:
        raise FileNotFoundError("No launcher jar found in JDT-LS plugins directory")
    jar = max(launcher_jars)

    # Step 4: Start the language server
    cmd = [
        "java",
        "-Declipse.application=org.eclipse.jdt.ls.core.id1",
        "-Dosgi.bundles.defaultStartLevel=4",
        "-Declipse.product=org.eclipse.jdt.ls.core.product",
        "-Dlog.protocol=true",
        "-Dlog.level=ERROR",
        "-Xmx2G",
        "-XX:+UseG1GC",
        "-jar", os.path.join(jdtls_path, "plugins", jar),
        "-configuration", config_dir,
        "-data", workspace_dir,
        "--add-modules=ALL-SYSTEM",
        "--add-opens", "java.base/java.util=ALL-UNNAMED",
        "--add-opens", "java.base/java.lang=ALL-UNNAMED"
    ]

    print(f"[DEBUG] Launching JDT LS...")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    threading.Thread(target=reader, args=(proc,), daemon=True).start()
    threading.Thread(target=stderr_reader, args=(proc,), daemon=True).start()

    # Step 5: Initialize LSP
    print("[DEBUG] Initializing...")
    java_home = "/Users/ameyd/Documents/apps/Android Studio.app/Contents/jre/Contents/Home"
    send(proc, {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "processId": os.getpid(),
            "clientInfo": {
                "name": "python-jdt-client",
                "version": "1.0.0"
            },
            "rootUri": f"file://{project_root}",
            "capabilities": {
                "textDocument": {
                    "publishDiagnostics": {
                        "relatedInformation": True
                    }
                }
            },
            "initializationOptions": {
                "workspaceFolders": [f"file://{project_root}"],
                "settings": {
                    "java": {
                        "home": java_home,
                        "import": {"maven": {"enabled": True}},
                        "maven": {"downloadSources": True}
                    }
                }
            },
            "workspaceFolders": [{
                "uri": f"file://{project_root}",
                "name": os.path.basename(project_root)
            }]
        }
    })

    time.sleep(2)
    send(proc, {"jsonrpc": "2.0", "method": "initialized", "params": {}})

    print("[DEBUG] Waiting for project import...")
    for i in range(45):
        if project_imported:
            break
        time.sleep(1)
        if i % 5 == 0:
            print(f"[DEBUG] Still waiting... ({i}s)")
    print("[DEBUG] Project import complete.")

    # Step 6: Open and analyze the Java file
    java_file = os.path.abspath(java_file)
    if not os.path.exists(java_file):
        print(f"[ERROR] Java file not found: {java_file}")
        proc.terminate()
        return []

    with open(java_file, "r") as f:
        text = f.read()

    send(proc, {
        "jsonrpc": "2.0",
        "method": "textDocument/didOpen",
        "params": {
            "textDocument": {
                "uri": f"file://{java_file}",
                "languageId": "java",
                "version": 1,
                "text": text
            }
        }
    })

    print("[DEBUG] Waiting for diagnostics...")
    time.sleep(10)

    send(proc, {"jsonrpc": "2.0", "id": 999, "method": "shutdown", "params": {}})
    time.sleep(1)
    send(proc, {"jsonrpc": "2.0", "method": "exit", "params": {}})
    proc.terminate()

    uri = f"file://{java_file}"
    file_errors = errors.get(uri, [])

    return [f"[Line {d['range']['start']['line']+1}] {d['message']}" for d in file_errors]

# === RUN SCRIPT ===
if __name__ == "__main__":
    project_root = "/Users/ameyd/Downloads/fitnodi"
    workspace_dir = "/tmp/.jdtls-workspace-fitnodi"  # FIXED: avoid overlapping workspace
    java_file = os.path.join(project_root, "src/main/java/com/asdsoft/fitnodi/controller/BaseController.java")

    jdtls_path = "/Users/ameyd/Downloads/jdt-language-server-1.49.0-202506271916"
    config_dir = os.path.join(jdtls_path, "config_mac_arm")

    errors = get_java_errors(
        java_file=java_file,
        project_root=project_root,
        jdtls_path=jdtls_path,
        config_dir=config_dir,
        workspace_dir=workspace_dir
    )

    if errors:
        print("\n=== ERRORS FOUND ===")
        for err in errors:
            print(err)
    else:
        print("✅ No errors found. File appears clean.")
