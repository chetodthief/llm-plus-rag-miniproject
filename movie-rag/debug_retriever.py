import sys

try:
    import langchain
    print(f"langchain version: {langchain.__version__}")
    print(f"langchain path: {langchain.__file__}")
except Exception as e:
    print(f"langchain import error: {e}")

try:
    from langchain.retrievers import EnsembleRetriever
    print("Successfully imported EnsembleRetriever from langchain.retrievers")
except Exception as e:
    print(f"EnsembleRetriever import error: {e}")

try:
    import pkgutil
    import importlib
    results = []
    for module_info in pkgutil.walk_packages(langchain.__path__, langchain.__name__ + '.'):
        if 'retriever' in module_info.name.lower():
            results.append(module_info.name)
    print("Found retriever modules: ", results)
except Exception as e:
    print(f"Module walk error: {e}")
