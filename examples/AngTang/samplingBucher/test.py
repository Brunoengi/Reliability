import os
import subprocess

# Pasta atual onde o script está localizado
folder_path = os.path.dirname(os.path.abspath(__file__))

# Lista todos os arquivos .py da pasta em ordem alfabética
scripts = sorted([f for f in os.listdir(folder_path) if f.endswith(".py")])

for script in scripts:
    # Ignora o próprio script para evitar rodar ele mesmo
    if script == os.path.basename(__file__):
        continue

    script_path = os.path.join(folder_path, script)
    print(f"Running {script}...")
    
    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print(f"Errors:\n{result.stderr}")
    
    print("-" * 40)