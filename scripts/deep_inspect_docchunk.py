"""
Deep introspection of DocChunk and DocMeta API

Verifies:
1. Does DocChunk expose token_count directly?
2. What exactly is in chunk.meta.model_dump()?
3. Do we need derived fields or is everything in docling_meta?
"""

import json
from docling.chunking import HybridChunker
from transformers import AutoTokenizer
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

print("=" * 80)
print("DEEP DOCCHUNK API INTROSPECTION")
print("=" * 80)
print()

# Create realistic test document
tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
chunker = HybridChunker(tokenizer=tokenizer)

doc = DoclingDocument(name="test_doc")
doc.add_heading(level=1, text="Introduction")
doc.add_text(label=DocItemLabel.PARAGRAPH, text="This is a paragraph with some content.")
doc.add_heading(level=2, text="Subsection")
doc.add_text(label=DocItemLabel.PARAGRAPH, text="Another paragraph here.")

chunks = list(chunker.chunk(doc))
chunk = chunks[0]

print("--- DocChunk Attributes ---")
print(f"Type: {type(chunk)}")
print(f"Module: {type(chunk).__module__}")
print()

# Check for token_count attribute
print("--- Checking for token_count ---")
if hasattr(chunk, "token_count"):
    print(f"✓ chunk.token_count exists: {chunk.token_count}")
else:
    print("✗ chunk.token_count does NOT exist")

if hasattr(chunk, "num_tokens"):
    print(f"✓ chunk.num_tokens exists: {chunk.num_tokens}")
else:
    print("✗ chunk.num_tokens does NOT exist")

if hasattr(chunk, "tokens"):
    print(f"✓ chunk.tokens exists: {chunk.tokens}")
else:
    print("✗ chunk.tokens does NOT exist")

print()

# Show all non-private attributes
print("--- All DocChunk attributes ---")
for attr in sorted(dir(chunk)):
    if not attr.startswith('_'):
        try:
            value = getattr(chunk, attr)
            if not callable(value):
                print(f"  {attr}: {type(value).__name__}")
        except Exception as e:
            print(f"  {attr}: <error: {e}>")
print()

# Test model_dump
print("--- DocChunk.model_dump() ---")
try:
    chunk_dump = chunk.model_dump()
    print("✓ chunk.model_dump() works")
    print(f"Keys: {list(chunk_dump.keys())}")
    print()
    print("Full dump:")
    print(json.dumps(chunk_dump, indent=2, default=str))
except Exception as e:
    print(f"✗ Error: {e}")

print()
print("=" * 80)

# Now inspect DocMeta
meta = chunk.meta
print("--- DocMeta Type ---")
print(f"Type: {type(meta)}")
print(f"Module: {type(meta).__module__}")
print()

print("--- All DocMeta attributes ---")
for attr in sorted(dir(meta)):
    if not attr.startswith('_'):
        try:
            value = getattr(meta, attr)
            if not callable(value):
                print(f"  {attr}: {type(value).__name__}")
        except Exception as e:
            print(f"  {attr}: <error: {e}>")
print()

# Test meta.model_dump
print("--- DocMeta.model_dump() ---")
try:
    meta_dump = meta.model_dump()
    print("✓ meta.model_dump() works")
    print(f"Keys: {list(meta_dump.keys())}")
    print()
    print("Full dump:")
    print(json.dumps(meta_dump, indent=2, default=str))
except Exception as e:
    print(f"✗ Error: {e}")

print()

# Check if derived fields are truly needed
print("--- Verification: Are derived fields in docling_meta? ---")
if meta_dump:
    print(f"headings in dump: {'headings' in meta_dump}")
    print(f"captions in dump: {'captions' in meta_dump}")
    print(f"page_numbers in dump: {'page_numbers' in meta_dump}")
    print(f"doc_items in dump: {'doc_items' in meta_dump}")
    print(f"bboxes in dump: {'bboxes' in meta_dump}")
    
    if 'headings' in meta_dump:
        print(f"  headings value: {meta_dump['headings']}")
    if 'page_numbers' in meta_dump:
        print(f"  page_numbers value: {meta_dump['page_numbers']}")

print()
print("=" * 80)
print("INTROSPECTION COMPLETE")
print("=" * 80)
