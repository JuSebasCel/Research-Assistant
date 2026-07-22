"""
Inspect Docling HybridChunker API

Discovers the real API of HybridChunker to implement chunking correctly.
"""

import inspect
from docling.chunking import HybridChunker

print("=" * 80)
print("DOCLING HYBRIDCHUNKER API INTROSPECTION")
print("=" * 80)
print()

# Inspect __init__
print("─" * 80)
print("HybridChunker.__init__ signature:")
print("─" * 80)
sig = inspect.signature(HybridChunker.__init__)
print(f"__init__{sig}")
print()

# Get parameters details
for param_name, param in sig.parameters.items():
    if param_name == 'self':
        continue
    default = param.default if param.default != inspect.Parameter.empty else "REQUIRED"
    annotation = param.annotation if param.annotation != inspect.Parameter.empty else "Any"
    print(f"  {param_name}: {annotation} = {default}")
print()

# Inspect all public methods
print("─" * 80)
print("Public Methods:")
print("─" * 80)
for name in dir(HybridChunker):
    if name.startswith('_'):
        continue
    attr = getattr(HybridChunker, name)
    if callable(attr):
        try:
            sig = inspect.signature(attr)
            print(f"  {name}{sig}")
        except Exception:
            print(f"  {name}(...)")
print()

# Inspect chunk result type
print("─" * 80)
print("Testing chunk() return type:")
print("─" * 80)
print("Create minimal chunker to inspect chunk type...")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")

chunker = HybridChunker(tokenizer=tokenizer)
print(f"✓ HybridChunker created")
print()

# Test with minimal doc
from docling_core.types.doc import DoclingDocument, TextItem
from docling_core.types.doc.labels import DocItemLabel

doc = DoclingDocument(name="test")
doc.add_text(label=DocItemLabel.PARAGRAPH, text="This is a test paragraph.")

chunks = list(chunker.chunk(doc))
print(f"✓ Generated {len(chunks)} chunks")
print()

if chunks:
    chunk = chunks[0]
    print(f"Chunk type: {type(chunk).__name__}")
    print(f"Chunk module: {type(chunk).__module__}")
    print()
    
    print("Chunk attributes:")
    for attr in dir(chunk):
        if not attr.startswith('_'):
            try:
                value = getattr(chunk, attr)
                if not callable(value):
                    print(f"  {attr}: {type(value).__name__}")
            except Exception:
                pass
    print()
    
    # Test contextualize
    print("─" * 80)
    print("Testing contextualize():")
    print("─" * 80)
    try:
        sig = inspect.signature(chunker.contextualize)
        print(f"contextualize{sig}")
        print()
        
        context_text = chunker.contextualize(chunk)
        print(f"✓ contextualize() returned: {type(context_text).__name__}")
        print(f"  Content: {context_text[:100]}...")
    except Exception as e:
        print(f"✗ Error: {e}")

print()
print("=" * 80)
print("INTROSPECTION COMPLETE")
print("=" * 80)
