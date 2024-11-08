import os

def classesInNamespace(namespace):

    l = ""
    cwd = os.getcwd().replace("\\", "/")
    modules = []

    for module in os.listdir(f"{cwd}/{namespace}"):
        if module[-3:] == ".py":
            l += f"from {namespace}.{module[:-3]} import *\n"
            modules.append(module[:-3])
    
    return l, modules

def addImportToFileAsFile(filename, namespace):
    import_text, modules = classesInNamespace(namespace)

    with open(filename, "r") as f:
        lines = f.readlines()
    with open(filename, "w") as f:
        f.write(import_text)

        for line in lines:
            if "from data_services" in line.strip("\n"):
                continue
            else:
                f.write(line)

if __name__ == "main":
    addImportToFileAsFile("src/temp_pipeline.py", "src/data_services")
