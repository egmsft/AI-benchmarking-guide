import json
import subprocess
import os
from Infra import tools
from prettytable import PrettyTable

class NVBandwidth:
    def __init__(self, path:str, machine: str):
        self.name='NVBandwidth'
        self.machine_name = machine

    def build(self):
        current = os.getcwd()
        path ='nvbandwidth'
        isdir = os.path.isdir(path)
        if not isdir:
            results = subprocess.run(['git', 'clone', 'https://github.com/NVIDIA/nvbandwidth', path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        build_path = os.path.join(current, 'nvbandwidth')
        os.chdir(build_path)
        results = subprocess.run(['sed', '-i', '2i\set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)', 'CMakeLists.txt'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if os.path.exists("/.dockerenv"):
            results = subprocess.run('apt update && ./debian_install.sh', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            results = subprocess.run('sudo apt update && sudo ./debian_install.sh', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        tools.write_log(tools.check_error(results))
        os.chdir(current)

    def run(self):
        current = os.getcwd()
        os.chdir(os.path.join(current, 'nvbandwidth'))
        print("Running NVBandwidth...")
        results = subprocess.run('./nvbandwidth -t device_to_host_memcpy_ce host_to_device_memcpy_ce device_to_device_bidirectional_memcpy_read_ce', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        tools.write_log(tools.check_error(results))
        log = results.stdout.decode('utf-8')
        os.chdir(current)

        item = "\n".join(log[log.find("device_to_host_memcpy_ce"):].strip().splitlines()[:-2])
        self.format_output(item)
        os.chdir(current)
        
    def format_output(self, output):
        tables, current = [], []
        for line in text.splitlines():
            if not line.strip():
                if current:
                    tables.append(current)
                    current = []
                continue
            if 'SUM' in line:
                continue
            row = [x.strip() for x in line.split() if x.strip()]
            current.append([float(x) if x.replace('.', '', 1).isdigit() else x for x in row])
        if current:
            tables.append(current)
        result = [tables[0], tables[1], tables[4]]
        labels = ["Device to Host memcpy", "Host to Device memcpy", "Device to Device Bidirectional memcpy Total"]
        for i in range(len(result)):
            result[i][0].insert(0, " ")
            t = PrettyTable(result[i][0])
            for j in range(1, len(result[i])):
                t.add_row(result[i][j])
            print(labels[i])
            print(t)
            tools.export_markdown("NV Bandwidth", labels[i], t)
            
