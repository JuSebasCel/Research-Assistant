"""
Inspect DocChunk and DocMeta API

Discovers complete API to properly serialize metadata.
"""

import inspect
from docling.chunking import HybridChunker
from transformers import AutoTokenizer
from docling_core.types.doc import DoclingDocument, TextItem
from docling_core.types.doc.labels import DocItemLabel

print("=" * 80)
print("DOCCHUNK AND DOCMETA API INTROSPECTION")
print("=" * 80)
print()

# Create test chunk
tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
chunker = HybridChunker(tokenizer=tokenizer)

doc = DoclingDocument(name="test")
doc.add_text(label=DocItemLabel.PARAGRAPH, text="This is a test paragraph.")

chunks = list(chunker.chunk(doc))
chunk = chunks[0]

print("-" * 80)
print("DocChunk Type:")
print("-" * 80)
print(f"Type: {type(chunk).__name__}")
print(f"Module: {type(chunk).__module__}")
print()

print("-" * 80)
print("DocChunk Attributes:")
print("-" * 80)
for attr in dir(chunk):
    if not attr.startswith('_'):
        try:
            value = getattr(chunk, attr)
            if not callable(value):
                print(f"  {attr}: {type(value).__name__}")
        except Exception:
            pass
print()

print("-" * 80)
print("DocChunk Serialization Methods:")
print("-" * 80)
serialization_methods = [
    "model_dump", "model_dump_json", "dict", "json", 
    "export_to_dict", "to_dict", "serialize"
]
for method in serialization_methods:
    if hasattr(chunk, method):
        try:
            sig = inspect.signature(getattr(chunk, method))
            print(f"✓ {method}{sig}")
        except Exception:
            print(f"✓ {method}(...)")
    else:
        print(f"✗ {method} - not found")
print()

# Test serialization
print("-" * 80)
print("Testing model_dump():")
print("-" * 80)
try:
    chunk_dict = chunk.model_dump()
    print(f"✓ model_dump() works")
    print(f"  Keys: {list(chunk_dict.keys())}")
    print(f"  Type: {type(chunk_dict)}")
    print()
    print("Full dump:")
    import json
    print(json.dumps(chunk_dict, indent=2, default=str))
except Exception as e:
    print(f"✗ Error: {e}")
print()

# Inspect DocMeta
print("-" * 80)
print("DocMeta Type:")
print("-" * 80)
meta = chunk.meta
print(f"Type: {type(meta).__name__}")
print(f"Module: {type(meta).__module__}")
print()

print("-" * 80)
print("DocMeta Attributes:")
print("-" * 80)
for attr in dir(meta):
    if not attr.startswith('_'):
        try:
            value = getattr(meta, attr)
            if not callable(value):
                value_repr = repr(value)[:60]
                print(f"  {attr}: {type(value).__name__} = {value_repr}")
        except Exception:
            pass
print()

print("-" * 80)
print("DocMeta Serialization Methods:")
print("-" * 80)
for method in serialization_methods:
    if hasattr(meta, method):
        try:
            sig = inspect.signature(getattr(meta, method))
            print(f"✓ {method}{sig}")
        except Exception:
            print(f"✓ {method}(...)")
    else:
        print(f"✗ {method} - not found")
print()

# Test meta serialization
print("-" * 80)
print("Testing meta.model_dump():")
print("-" * 80)
try:
    meta_dict = meta.model_dump()
    print(f"✓ meta.model_dump() works")
    print(f"  Keys: {list(meta_dict.keys())}")
    print()
    print("Full meta dump:")
    print(json.dumps(meta_dict, indent=2, default=str))
except Exception as e:
    print(f"✗ Error: {e}")

print()
print("=" * 80)
print("INTROSPECTION COMPLETE")
print("=" * 80)
