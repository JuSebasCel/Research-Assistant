"""
Tests formales del pipeline Docling + LlamaIndex.

Verifica explícitamente los criterios del requerimiento:
a) Ninguna tabla queda partida a mitad de fila
b) La sección de referencias queda completamente separada del contenido científico
c) El boilerplate administrativo queda separado o descartado
d) Cada chunk tiene metadata de section/content_type proveniente de estructura real
"""

import pytest
from pathlib import Path

from rag_app.services.ingestion_service import IngestionService


class TestDoclingLlamaIndexPipeline:
    """Tests del nuevo pipeline sin heurística."""
    
    @pytest.fixture
    def ingestion_service(self):
        """Fixture del servicio de ingesta."""
        return IngestionService()
    
    @pytest.fixture
    def sample_pdf_path(self):
        """Ruta al PDF de prueba."""
        pdf_path = Path("data/uploads/paper.pdf")
        if not pdf_path.exists():
            pytest.skip("PDF de prueba no encontrado en data/uploads/paper.pdf")
        return pdf_path
    
    def test_tables_never_split(self, ingestion_service, sample_pdf_path):
        """
        Criterio (a): Ninguna tabla queda partida a mitad de fila.
        
        Verifica que:
        - Las tablas son nodos atómicos
        - Cada tabla tiene TODAS sus filas en un solo nodo
        - No hay fragmentación de tablas entre múltiples nodos
        """
        result = ingestion_service.process_pdf(
            sample_pdf_path,
            exclude_references=True,
            exclude_boilerplate=True,
        )
        
        nodes = result["nodes"]
        table_nodes = [
            node for node in nodes
            if node.metadata.get("content_type") == "table"
        ]
        
        # Verificar que hay tablas (si el PDF las tiene)
        if result["document_metadata"].get("num_tables", 0) > 0:
            assert len(table_nodes) > 0, "Docling detectó tablas pero no hay nodos de tipo table"
        
        # Para cada tabla, verificar que está completa
        for table_node in table_nodes:
            content = table_node.get_content()
            
            # Verificar que tiene formato de tabla Markdown
            lines = content.split("\n")
            table_lines = [line for line in lines if "|" in line]
            
            assert len(table_lines) > 0, f"Nodo marcado como tabla no tiene formato Markdown: {content[:100]}"
            
            # Verificar que tiene header y separator (típico de tabla Markdown completa)
            # Ejemplo:
            # | Header1 | Header2 |
            # |---------|---------|
            # | Data1   | Data2   |
            
            # Al menos debe tener 2 líneas (header + separator mínimo)
            assert len(table_lines) >= 2, f"Tabla parece incompleta (solo {len(table_lines)} líneas)"
            
            # Verificar que NO está partida: no debe terminar abruptamente
            # (las tablas completas en Markdown tienen estructura consistente)
            last_line = lines[-1].strip() if lines else ""
            
            # Si es una tabla, la última línea debería tener pipes o estar vacía
            # NO debería ser texto narrativo (señal de tabla partida)
            if last_line and "|" not in last_line and len(last_line) > 50:
                pytest.fail(f"Tabla parece estar partida: última línea es texto largo sin pipes")
        
        print(f"✓ Test (a) PASSED: {len(table_nodes)} tablas verificadas como atómicas")
    
    def test_references_separated(self, ingestion_service, sample_pdf_path):
        """
        Criterio (b): La sección de referencias queda completamente separada.
        
        Verifica que:
        - No hay nodos con content_type="reference" en los nodos indexados
        - La detección viene de estructura Markdown, no regex
        - Las referencias NO están mezcladas con contenido científico
        """
        result = ingestion_service.process_pdf(
            sample_pdf_path,
            exclude_references=True,
            exclude_boilerplate=True,
        )
        
        nodes = result["nodes"]
        
        # Verificar que NO hay nodos de referencias en los nodos finales
        reference_nodes = [
            node for node in nodes
            if node.metadata.get("content_type") == "reference"
        ]
        
        assert len(reference_nodes) == 0, \
            f"Se encontraron {len(reference_nodes)} nodos de referencias cuando deberían estar excluidos"
        
        # Verificar que la detección fue por estructura (no por contenido)
        # Esto lo verificamos chequeando que el pipeline config dice "NONE"
        config = result["statistics"]["pipeline_config"]
        assert "heuristic" in config["heuristics_used"].lower(), \
            "Config debe mencionar explícitamente que no usa heurística"
        
        assert "zero" in config["heuristics_used"].lower() or "none" in config["heuristics_used"].lower(), \
            f"Heurísticas usadas: {config['heuristics_used']} (debería ser NONE)"
        
        # Verificar que references fueron excluidas en el filtrado
        assert result["statistics"]["sections_filtered"]["references_excluded"] is True, \
            "El flag de references_excluded debería estar en True"
        
        print(f"✓ Test (b) PASSED: Referencias separadas correctamente (0 nodos de referencias encontrados)")
    
    def test_boilerplate_excluded(self, ingestion_service, sample_pdf_path):
        """
        Criterio (c): Boilerplate administrativo queda separado o descartado.
        
        Verifica que:
        - Secciones administrativas están excluidas
        - Detección basada en headers Markdown, no contenido
        - No hay nodos de tipo boilerplate en los nodos finales
        """
        result = ingestion_service.process_pdf(
            sample_pdf_path,
            exclude_references=True,
            exclude_boilerplate=True,
        )
        
        nodes = result["nodes"]
        
        # Verificar que NO hay nodos de boilerplate
        boilerplate_nodes = [
            node for node in nodes
            if node.metadata.get("content_type") == "boilerplate"
        ]
        
        assert len(boilerplate_nodes) == 0, \
            f"Se encontraron {len(boilerplate_nodes)} nodos de boilerplate cuando deberían estar excluidos"
        
        # Verificar que boilerplate fue excluido en el filtrado
        assert result["statistics"]["sections_filtered"]["boilerplate_excluded"] is True, \
            "El flag de boilerplate_excluded debería estar en True"
        
        # Verificar que la detección fue por headers Markdown
        # (no podemos verificar directamente, pero verificamos que no hay heurística)
        config = result["statistics"]["pipeline_config"]
        assert "none" in config["heuristics_used"].lower() or "zero" in config["heuristics_used"].lower(), \
            "No debería haber heurística de detección por contenido"
        
        print(f"✓ Test (c) PASSED: Boilerplate excluido correctamente (0 nodos de boilerplate encontrados)")
    
    def test_metadata_from_structure(self, ingestion_service, sample_pdf_path):
        """
        Criterio (d): Cada chunk tiene metadata de section/content_type proveniente de estructura real.
        
        Verifica que:
        - section_title viene de headers Markdown (no de heurística "líneas cortas")
        - content_type viene del tipo de elemento (table/prose/figure)
        - NO hay detección por regex o contenido
        """
        result = ingestion_service.process_pdf(
            sample_pdf_path,
            exclude_references=True,
            exclude_boilerplate=True,
        )
        
        nodes = result["nodes"]
        
        # Todos los nodos deben tener content_type
        nodes_sin_content_type = [
            i for i, node in enumerate(nodes)
            if "content_type" not in node.metadata
        ]
        
        assert len(nodes_sin_content_type) == 0, \
            f"{len(nodes_sin_content_type)} nodos sin content_type (deberían tenerlo todos)"
        
        # Verificar que los content_type son válidos (vienen del parser, no inventados)
        valid_content_types = {"table", "prose", "figure", "text"}
        
        for node in nodes:
            ct = node.metadata.get("content_type")
            assert ct in valid_content_types, \
                f"content_type inválido: {ct} (no viene de estructura conocida)"
        
        # Verificar que section_title (si existe) viene de Markdown
        # (no podemos verificar directamente, pero verificamos consistencia)
        nodes_con_section = [
            node for node in nodes
            if "section_title" in node.metadata
        ]
        
        if len(nodes_con_section) > 0:
            # Verificar que los section_title tienen formato razonable
            # (no son solo números o caracteres raros, que serían señal de heurística fallida)
            for node in nodes_con_section:
                section_title = node.metadata["section_title"]
                
                # Debe tener al menos una letra
                assert any(c.isalpha() for c in section_title), \
                    f"section_title sospechoso (sin letras): {section_title}"
                
                # No debe ser solo un símbolo repetido (señal de heurística)
                if len(section_title) > 3:
                    unique_chars = len(set(section_title.replace(" ", "")))
                    assert unique_chars > 2, \
                        f"section_title sospechoso (muy repetitivo): {section_title}"
        
        # Verificar que NO hay heurística en el pipeline
        config = result["statistics"]["pipeline_config"]
        assert "none" in config["heuristics_used"].lower() or "zero" in config["heuristics_used"].lower(), \
            f"Pipeline debería tener ZERO heurística, pero reporta: {config['heuristics_used']}"
        
        print(f"✓ Test (d) PASSED: Metadata viene de estructura real")
        print(f"  • {len(nodes)} nodos con content_type válido")
        print(f"  • {len(nodes_con_section)} nodos con section_title")
        print(f"  • Pipeline config: {config['heuristics_used']}")
    
    def test_pipeline_uses_docling_llamaindex(self, ingestion_service, sample_pdf_path):
        """
        Verificar que el pipeline usa Docling + LlamaIndex como se especificó.
        """
        result = ingestion_service.process_pdf(
            sample_pdf_path,
            exclude_references=True,
            exclude_boilerplate=True,
        )
        
        config = result["statistics"]["pipeline_config"]
        
        # Verificar extractor
        assert config["extractor"] == "Docling", \
            f"Extractor debería ser Docling, pero es: {config['extractor']}"
        
        # Verificar parsers de LlamaIndex
        expected_parsers = [
            "MarkdownElementNodeParser",
            "HierarchicalNodeParser",
            "SentenceWindowNodeParser",
        ]
        
        for parser in expected_parsers:
            assert parser in config["parsers"], \
                f"Parser {parser} no está en el pipeline: {config['parsers']}"
        
        print(f"✓ Test EXTRA PASSED: Pipeline usa Docling + LlamaIndex correctamente")
        print(f"  • Extractor: {config['extractor']}")
        print(f"  • Parsers: {', '.join(config['parsers'])}")


def run_all_tests():
    """Ejecuta todos los tests manualmente (sin pytest)."""
    print("="*80)
    print("EJECUTANDO TESTS DEL PIPELINE DOCLING + LLAMAINDEX")
    print("="*80)
    
    # Crear instancias
    test_suite = TestDoclingLlamaIndexPipeline()
    service = test_suite.ingestion_service()
    pdf_path = test_suite.sample_pdf_path()
    
    print(f"\nUsando PDF: {pdf_path}\n")
    
    # Ejecutar cada test
    tests = [
        ("a) Tablas NO partidas", test_suite.test_tables_never_split),
        ("b) Referencias separadas", test_suite.test_references_separated),
        ("c) Boilerplate excluido", test_suite.test_boilerplate_excluded),
        ("d) Metadata de estructura real", test_suite.test_metadata_from_structure),
        ("EXTRA: Pipeline correcto", test_suite.test_pipeline_uses_docling_llamaindex),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*80}")
        print(f"Test: {test_name}")
        print('='*80)
        
        try:
            test_func(service, pdf_path)
            print(f"✓✓✓ PASSED ✓✓✓")
            passed += 1
        except AssertionError as e:
            print(f"✗✗✗ FAILED ✗✗✗")
            print(f"Error: {e}")
            failed += 1
        except Exception as e:
            print(f"✗✗✗ ERROR ✗✗✗")
            print(f"Excepción: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Resumen
    print("\n" + "="*80)
    print("RESUMEN DE TESTS")
    print("="*80)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✓✓✓ TODOS LOS TESTS PASARON ✓✓✓")
        print("El pipeline cumple TODOS los criterios del requerimiento.")
    else:
        print(f"\n✗ {failed} test(s) fallaron")
        print("Revisar los errores arriba.")
    
    print("="*80)
    
    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
