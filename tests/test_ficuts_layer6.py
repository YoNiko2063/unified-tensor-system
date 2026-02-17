"""
Tests for FICUTS Layer 6: Web Ingestion
"""

import pytest
from tensor.web_ingestion import ArticleParser, ResearchConceptExtractor


SAMPLE_HTML = """
<html>
    <title>Test Article</title>
    <body>
        <h1>Main Title</h1>
        <h2>Section 1</h2>
        <p>Content with <b>bold</b> and <a href="http://example.com">link</a></p>
        <pre>code block</pre>
        <h2>Section 2</h2>
        <p>More content with <em>italic</em> and <a href="#local">local link</a></p>
        <blockquote>A quoted passage</blockquote>
        <code>inline code</code>
    </body>
</html>
"""


def test_article_parser_title():
    parser = ArticleParser()
    result = parser.parse(SAMPLE_HTML, 'http://test.com')
    assert result['title'] == 'Test Article'


def test_article_parser_sections():
    parser = ArticleParser()
    result = parser.parse(SAMPLE_HTML, 'http://test.com')
    assert len(result['sections']) == 3  # h1 + 2×h2
    assert result['sections'][0]['level'] == 1
    assert result['sections'][0]['text'] == 'Main Title'
    assert result['sections'][1]['level'] == 2
    assert result['sections'][1]['text'] == 'Section 1'


def test_article_parser_hyperlinks():
    parser = ArticleParser()
    result = parser.parse(SAMPLE_HTML, 'http://test.com')
    assert len(result['hyperlinks']) == 2
    outbound = [lk for lk in result['hyperlinks'] if lk['outbound']]
    local = [lk for lk in result['hyperlinks'] if not lk['outbound']]
    assert len(outbound) == 1
    assert outbound[0]['href'] == 'http://example.com'
    assert len(local) == 1


def test_article_parser_code_blocks():
    parser = ArticleParser()
    result = parser.parse(SAMPLE_HTML, 'http://test.com')
    assert len(result['code_blocks']) > 0
    assert 'code block' in result['code_blocks']


def test_article_parser_emphasis():
    parser = ArticleParser()
    result = parser.parse(SAMPLE_HTML, 'http://test.com')
    assert result['emphasis_map']['bold_count'] > 0
    assert result['emphasis_map']['italic_count'] > 0
    assert result['emphasis_map']['quote_count'] > 0


def test_article_parser_dom_depth():
    parser = ArticleParser()
    result = parser.parse(SAMPLE_HTML, 'http://test.com')
    assert result['dom_depth'] > 0


def test_article_parser_url_preserved():
    parser = ArticleParser()
    url = 'http://test.com/some/path'
    result = parser.parse(SAMPLE_HTML, url)
    assert result['url'] == url


def test_article_parser_no_title_fallback():
    """Falls back to <h1> if no <title> tag"""
    html = '<html><body><h1>Only H1</h1></body></html>'
    parser = ArticleParser()
    result = parser.parse(html, 'http://test.com')
    assert result['title'] == 'Only H1'


def test_article_parser_no_title_untitled():
    """Returns 'Untitled' when neither <title> nor <h1> present"""
    html = '<html><body><p>no heading</p></body></html>'
    parser = ArticleParser()
    result = parser.parse(html, 'http://test.com')
    assert result['title'] == 'Untitled'


def test_article_parser_empty_html():
    """Does not crash on empty document"""
    parser = ArticleParser()
    result = parser.parse('', 'http://test.com')
    assert result['title'] == 'Untitled'
    assert result['sections'] == []
    assert result['hyperlinks'] == []
    assert result['code_blocks'] == []


# ── Task 6.2: ResearchConceptExtractor ────────────────────────────────────────

CONCEPT_ARTICLE = {
    'sections': [
        {'text': 'We use exponential decay with tau = 5ms in our Neural Network model.'},
        {'text': 'The learning rate alpha = 0.01 was chosen empirically.'},
        {'text': 'Fisher Information Matrix helps bound parameter uncertainty.'},
    ],
    'code_blocks': [
        r'\frac{dV}{dt} = -\frac{V}{\tau}',
        r'y = A e^{-\lambda t}',
        'plain code without latex',
    ],
}


def test_concept_extractor_equations():
    extractor = ResearchConceptExtractor()
    result = extractor.extract(CONCEPT_ARTICLE)
    # Both LaTeX blocks should be captured; plain code should not
    assert len(result['equations']) == 2
    assert any('frac' in eq for eq in result['equations'])
    assert any('lambda' in eq for eq in result['equations'])


def test_concept_extractor_parameters():
    extractor = ResearchConceptExtractor()
    result = extractor.extract(CONCEPT_ARTICLE)
    assert len(result['parameters']) >= 2
    symbols = [p['symbol'] for p in result['parameters']]
    # 'tau' or 'alpha' (case-insensitive match) must be among them
    lowered = [s.lower() for s in symbols]
    assert 'tau' in lowered or 'alpha' in lowered


def test_concept_extractor_parameter_values():
    extractor = ResearchConceptExtractor()
    result = extractor.extract(CONCEPT_ARTICLE)
    values = {p['symbol'].lower(): p['value'] for p in result['parameters']}
    if 'tau' in values:
        assert values['tau'] == 5.0
    if 'alpha' in values:
        assert values['alpha'] == 0.01


def test_concept_extractor_technical_terms():
    extractor = ResearchConceptExtractor()
    result = extractor.extract(CONCEPT_ARTICLE)
    assert len(result['technical_terms']) > 0
    # "Neural Network" and "Fisher Information Matrix" are Title Case
    joined = ' '.join(result['technical_terms'])
    assert 'Neural Network' in joined or 'Fisher Information' in joined


def test_concept_extractor_has_experiment_false():
    extractor = ResearchConceptExtractor()
    result = extractor.extract(CONCEPT_ARTICLE)
    # No experimental indicators in CONCEPT_ARTICLE
    assert result['has_experiment'] is False


def test_concept_extractor_has_experiment_true():
    extractor = ResearchConceptExtractor()
    article = {
        'sections': [{'text': 'We measured the voltage across each node.'}],
        'code_blocks': [],
    }
    result = extractor.extract(article)
    assert result['has_experiment'] is True


def test_concept_extractor_no_equations():
    extractor = ResearchConceptExtractor()
    result = extractor.extract({'sections': [], 'code_blocks': ['just text']})
    assert result['equations'] == []


def test_concept_extractor_no_params():
    extractor = ResearchConceptExtractor()
    result = extractor.extract({'sections': [{'text': 'No assignments here.'}], 'code_blocks': []})
    assert result['parameters'] == []


def test_concept_extractor_deduplicates_terms():
    extractor = ResearchConceptExtractor()
    article = {
        'sections': [
            {'text': 'Fisher Information Matrix is great. Fisher Information Matrix again.'},
        ],
        'code_blocks': [],
    }
    result = extractor.extract(article)
    terms = result['technical_terms']
    # set-dedup should produce only one entry for "Fisher Information Matrix"
    assert terms.count('Fisher Information Matrix') <= 1
