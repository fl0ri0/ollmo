import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ollmo_services.response_artifact_bundles import bundle_response_artifacts


class ResponseArtifactBundleTests(unittest.TestCase):
    def test_web_bundle_rewrites_html_and_css_links(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            document_dir = root / 'artifacts' / 'documents'
            image_dir = root / 'artifacts' / 'images'
            document_dir.mkdir(parents=True)
            image_dir.mkdir(parents=True)
            index_path = document_dir / 'index.html'
            styles_path = document_dir / 'styles.css'
            image_path = image_dir / 'hero.png'
            index_original = (
                '<!doctype html><link rel="stylesheet" href="styles.css">'
                '<div class="hero" style="background-image: url(\'../images/hero.png\');"></div>'
                '<img src="../images/hero.png">'
                '<source srcset="../images/hero.png 1x, ../images/hero.png 2x">'
            )
            styles_original = '.hero { background-image: url("../images/hero.png"); }'
            index_path.write_text(index_original, encoding='utf-8')
            styles_path.write_text(styles_original, encoding='utf-8')
            image_path.write_bytes(b'png')

            payload = bundle_response_artifacts(
                {
                    'id': 'resp_customer_demo',
                    'artifacts': [
                        {'type': 'text', 'path': str(index_path), 'name': 'index', 'artifact_ref': 'artifact:index'},
                        {'type': 'text', 'path': str(styles_path), 'name': 'styles', 'artifact_ref': 'artifact:styles'},
                        {'type': 'image', 'path': str(image_path), 'name': 'hero', 'artifact_ref': 'artifact:hero'},
                    ],
                },
                target_name='client',
                bundle_root=root / 'bundles',
                created_at='2026-05-26T10:30:00Z',
            )

            bundle_dir = Path(payload['bundle_path'])
            bundled_index = bundle_dir / 'index.html'
            bundled_styles = bundle_dir / 'assets' / 'css' / 'styles.css'
            bundled_image = bundle_dir / 'assets' / 'images' / 'hero.png'

            self.assertEqual(payload['status'], 'bundled')
            self.assertTrue(bundled_index.exists())
            self.assertTrue(bundled_styles.exists())
            self.assertTrue(bundled_image.exists())
            bundled_html = bundled_index.read_text(encoding='utf-8')
            self.assertIn('href="assets/css/styles.css"', bundled_html)
            self.assertIn("url('assets/images/hero.png')", bundled_html)
            self.assertIn('src="assets/images/hero.png"', bundled_html)
            self.assertIn('assets/images/hero.png 1x', bundled_html)
            self.assertIn('url("../images/hero.png")', bundled_styles.read_text(encoding='utf-8'))
            self.assertEqual(payload['link_check']['status'], 'passed')
            self.assertIn('html_style_url', {item['kind'] for item in payload['rewritten_links']})
            self.assertEqual(index_path.read_text(encoding='utf-8'), index_original)
            self.assertEqual(styles_path.read_text(encoding='utf-8'), styles_original)
            for artifact in payload['copied_artifacts']:
                final_bytes = Path(artifact['path']).read_bytes()
                self.assertEqual(artifact['size_bytes'], len(final_bytes))
                self.assertEqual(artifact['sha256'], hashlib.sha256(final_bytes).hexdigest())

    def test_web_bundle_rewrites_static_html_fetch_and_preserves_nonlocal_or_dynamic_fetches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            document_dir = root / 'artifacts' / 'documents'
            document_dir.mkdir(parents=True)
            index_path = document_dir / 'index.html'
            pricing_path = document_dir / 'pricing.json'
            index_original = (
                '<!doctype html><script>'
                'fetch("pricing.json");'
                'fetch("https://example.com/pricing.json");'
                'fetch("data:application/json,%7B%7D");'
                'fetch("/api/pricing.json");'
                'fetch("/api/pricing");'
                'fetch("dynamic.json" + window.location.search);'
                'const dynamicPath = window.pricingPath; fetch(dynamicPath);'
                '</script>'
            )
            index_path.write_text(index_original, encoding='utf-8')
            pricing_path.write_text('{"materials": []}\n', encoding='utf-8')

            payload = bundle_response_artifacts(
                {
                    'id': 'resp_static_html_fetch',
                    'artifacts': [
                        {
                            'type': 'text',
                            'path': str(index_path),
                            'name': 'index',
                            'artifact_ref': 'artifact:index',
                        },
                        {
                            'type': 'text',
                            'path': str(pricing_path),
                            'name': 'pricing',
                            'artifact_ref': 'artifact:pricing',
                        },
                    ],
                },
                bundle_root=root / 'bundles',
                created_at='2026-08-15T13:30:00Z',
            )

            bundle_dir = Path(payload['bundle_path'])
            bundled_html = (bundle_dir / 'index.html').read_text(encoding='utf-8')
            self.assertEqual(payload['status'], 'bundled')
            self.assertEqual(payload['link_check']['status'], 'passed')
            self.assertTrue((bundle_dir / 'assets/files/pricing.json').exists())
            self.assertIn('fetch("assets/files/pricing.json")', bundled_html)
            self.assertIn('fetch("https://example.com/pricing.json")', bundled_html)
            self.assertIn('fetch("data:application/json,%7B%7D")', bundled_html)
            self.assertIn('fetch("/api/pricing.json")', bundled_html)
            self.assertIn('fetch("/api/pricing")', bundled_html)
            self.assertIn('fetch("dynamic.json" + window.location.search)', bundled_html)
            self.assertIn('fetch(dynamicPath)', bundled_html)
            self.assertIn(
                'html_fetch',
                {item.get('kind') for item in payload['rewritten_links']},
            )
            self.assertEqual(index_path.read_text(encoding='utf-8'), index_original)

    def test_web_bundle_fails_link_check_for_missing_static_js_fetch_target(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            document_dir = root / 'artifacts' / 'documents'
            document_dir.mkdir(parents=True)
            index_path = document_dir / 'index.html'
            script_path = document_dir / 'app.js'
            index_path.write_text(
                '<!doctype html><script src="app.js"></script>',
                encoding='utf-8',
            )
            script_path.write_text(
                "fetch('pricing.json');\n"
                "fetch('https://example.com/pricing.json');\n"
                "fetch('data:application/json,%7B%7D');\n"
                "fetch('/api/pricing.json');\n"
                "fetch('/api/pricing');\n"
                "fetch('dynamic.json' + window.location.search);\n"
                'const dynamicPath = window.pricingPath; fetch(dynamicPath);\n',
                encoding='utf-8',
            )

            payload = bundle_response_artifacts(
                {
                    'id': 'resp_missing_static_js_fetch',
                    'outputs': [
                        {
                            'type': 'text',
                            'status': 'fulfilled',
                            'artifact_ref': 'artifact:index',
                        }
                    ],
                    'artifacts': [
                        {
                            'type': 'text',
                            'path': str(index_path),
                            'name': 'index',
                            'artifact_ref': 'artifact:index',
                        },
                        {
                            'type': 'text',
                            'path': str(script_path),
                            'name': 'app',
                            'artifact_ref': 'artifact:app',
                        },
                    ],
                },
                bundle_root=root / 'bundles',
                created_at='2026-08-15T13:35:00Z',
            )

            self.assertEqual(payload['status'], 'failed')
            self.assertEqual(payload['link_check']['status'], 'failed')
            self.assertEqual(
                [
                    (item.get('kind'), item.get('target'))
                    for item in payload['link_check']['missing']
                    if item.get('reason') == 'missing'
                ],
                [('js_fetch', 'pricing.json')],
            )

    def test_bundle_rewrites_structural_json_asset_paths_and_includes_the_dependency(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            document_dir = root / 'artifacts' / 'documents'
            image_dir = root / 'artifacts' / 'images'
            document_dir.mkdir(parents=True)
            image_dir.mkdir(parents=True)
            pricing_path = document_dir / 'pricing.json'
            image_path = image_dir / 'hero.png'
            image_path.write_bytes(b'png')
            pricing_path.write_text(
                json.dumps(
                    {
                        'items': [
                            {
                                'material': 'Titanium',
                                'image_path': '../images/hero.png',
                                'description': '../images/not-a-link.png',
                            }
                        ],
                        'imagePaths': [str(image_path)],
                        'catalogUrl': 'https://example.com/catalog.json',
                    },
                    indent=2,
                ),
                encoding='utf-8',
            )

            payload = bundle_response_artifacts(
                {
                    'id': 'resp_json_paths',
                    'outputs': [
                        {
                            'type': 'text',
                            'status': 'fulfilled',
                            'artifact_ref': 'artifact:pricing',
                        }
                    ],
                    'artifacts': [
                        {
                            'type': 'text',
                            'path': str(pricing_path),
                            'name': 'pricing',
                            'artifact_ref': 'artifact:pricing',
                        }
                    ],
                },
                bundle_root=root / 'bundles',
                created_at='2026-08-04T19:00:00Z',
            )

            bundled_pricing = Path(payload['bundle_path']) / 'pricing.json'
            bundled_payload = json.loads(bundled_pricing.read_text(encoding='utf-8'))
            self.assertEqual(payload['status'], 'bundled')
            self.assertEqual(bundled_payload['items'][0]['image_path'], 'assets/images/hero.png')
            self.assertEqual(bundled_payload['imagePaths'][0], 'assets/images/hero.png')
            self.assertEqual(
                bundled_payload['items'][0]['description'],
                '../images/not-a-link.png',
            )
            self.assertEqual(bundled_payload['catalogUrl'], 'https://example.com/catalog.json')
            self.assertTrue((Path(payload['bundle_path']) / 'assets/images/hero.png').exists())
            json_rewrites = [
                item for item in payload['rewritten_links'] if item.get('kind') == 'json_path'
            ]
            self.assertEqual(
                {item.get('json_key_path') for item in json_rewrites},
                {'items[0].image_path', 'imagePaths[0]'},
            )

    def test_bundle_fails_closed_for_unresolved_structural_json_asset_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            document_dir = root / 'artifacts' / 'documents'
            document_dir.mkdir(parents=True)
            index_path = document_dir / 'index.html'
            pricing_path = document_dir / 'pricing.json'
            index_path.write_text('<!doctype html><main>Catalog</main>', encoding='utf-8')
            pricing_path.write_text(
                json.dumps(
                    {
                        'items': [
                            {
                                'image_path': '{{PATH_TITANIUM}}',
                                'thumbnailUrl': 'images/missing.jpg',
                                'description': 'images/not-a-structural-link.jpg',
                                'note': '{{NOT_A_PATH_VALUE}}',
                            }
                        ]
                    },
                    indent=2,
                ),
                encoding='utf-8',
            )

            payload = bundle_response_artifacts(
                {
                    'id': 'resp_json_missing_paths',
                    'artifacts': [
                        {
                            'type': 'text',
                            'path': str(index_path),
                            'name': 'index',
                            'artifact_ref': 'artifact:index',
                        },
                        {
                            'type': 'text',
                            'path': str(pricing_path),
                            'name': 'pricing',
                            'artifact_ref': 'artifact:pricing',
                        },
                    ],
                },
                bundle_root=root / 'bundles',
                created_at='2026-08-04T19:05:00Z',
            )

            self.assertEqual(payload['status'], 'failed')
            json_missing = [
                item
                for item in payload['link_check']['missing']
                if item.get('kind') == 'json_path'
            ]
            self.assertEqual(
                {item.get('target') for item in json_missing},
                {'{{PATH_TITANIUM}}', 'images/missing.jpg'},
            )
            self.assertEqual(
                {item.get('json_key_path') for item in json_missing},
                {'items[0].image_path', 'items[0].thumbnailUrl'},
            )
            self.assertNotIn(
                'images/not-a-structural-link.jpg',
                {item.get('target') for item in payload['link_check']['missing']},
            )

    def test_duplicate_filenames_are_deduped_in_bundle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first_dir = root / 'one'
            second_dir = root / 'two'
            first_dir.mkdir()
            second_dir.mkdir()
            first_image = first_dir / 'hero.png'
            second_image = second_dir / 'hero.png'
            first_image.write_bytes(b'one')
            second_image.write_bytes(b'two')

            payload = bundle_response_artifacts(
                {
                    'id': 'resp_dupes',
                    'artifacts': [
                        {'type': 'image', 'path': str(first_image), 'artifact_ref': 'artifact:hero_one'},
                        {'type': 'image', 'path': str(second_image), 'artifact_ref': 'artifact:hero_two'},
                    ],
                },
                bundle_root=root / 'bundles',
                created_at='2026-05-26T10:30:00Z',
            )

            relative_paths = [item['relative_path'] for item in payload['copied_artifacts']]
            self.assertEqual(sorted(relative_paths), ['assets/images/hero.png', 'assets/images/hero_2.png'])

    def test_external_links_are_untouched_and_missing_local_links_are_reported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            index_path = root / 'index.html'
            index_path.write_text(
                '<!doctype html><a href="https://example.com">External</a>'
                '<img src="data:image/png;base64,abc">'
                '<script src="missing.js"></script>',
                encoding='utf-8',
            )

            payload = bundle_response_artifacts(
                {
                    'id': 'resp_missing_link',
                    'artifacts': [
                        {'type': 'text', 'path': str(index_path), 'name': 'index', 'artifact_ref': 'artifact:index'},
                    ],
                },
                bundle_root=root / 'bundles',
                created_at='2026-05-26T10:30:00Z',
            )

            bundled_index = Path(payload['entrypoint'])
            html = bundled_index.read_text(encoding='utf-8')
            self.assertEqual(payload['status'], 'failed')
            self.assertIn('href="https://example.com"', html)
            self.assertIn('src="data:image/png;base64,abc"', html)
            self.assertEqual(payload['link_check']['status'], 'failed')
            self.assertEqual(payload['link_check']['missing'][0]['target'], 'missing.js')
            self.assertTrue(Path(payload['manifest_path']).exists())

    def test_malformed_html_syntax_fails_bundle_link_check(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            document_dir = root / 'artifacts' / 'documents'
            document_dir.mkdir(parents=True)
            index_path = document_dir / 'index.html'
            styles_path = document_dir / 'styles.css'
            index_path.write_text(
                '<!doctype html><html><head>'
                '\'link\' rel="stylesheet" href="styles.css">'
                '</head><body><section><<div class="icon">✨</div></section></body></html>',
                encoding='utf-8',
            )
            styles_path.write_text('body { color: white; }', encoding='utf-8')

            payload = bundle_response_artifacts(
                {
                    'id': 'resp_malformed_html',
                    'artifacts': [
                        {'type': 'text', 'path': str(index_path), 'name': 'index', 'artifact_ref': 'artifact:index'},
                        {'type': 'text', 'path': str(styles_path), 'name': 'styles', 'artifact_ref': 'artifact:styles'},
                    ],
                },
                bundle_root=root / 'bundles',
                created_at='2026-06-07T13:35:00Z',
            )

            self.assertEqual(payload['status'], 'failed')
            self.assertEqual(payload['link_check']['status'], 'failed')
            syntax_items = [
                item for item in payload['link_check']['missing']
                if item.get('reason') == 'syntax'
            ]
            self.assertTrue(syntax_items)
            joined_issues = '\n'.join(syntax_items[0].get('issues') or [])
            self.assertIn('quoted instead of angle-bracketed', joined_issues)

    def test_malformed_stylesheet_link_attrs_fail_bundle_link_check(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            document_dir = root / 'artifacts' / 'documents'
            document_dir.mkdir(parents=True)
            index_path = document_dir / 'index.html'
            styles_path = document_dir / 'styles.css'
            index_path.write_text(
                '<!doctype html><html><head>'
                '<link rel="stylesheet="href="styles.css">'
                '</head><body><main></main></body></html>',
                encoding='utf-8',
            )
            styles_path.write_text('body { color: white; }', encoding='utf-8')

            payload = bundle_response_artifacts(
                {
                    'id': 'resp_malformed_stylesheet_link_attrs',
                    'artifacts': [
                        {'type': 'text', 'path': str(index_path), 'name': 'index', 'artifact_ref': 'artifact:index'},
                        {'type': 'text', 'path': str(styles_path), 'name': 'styles', 'artifact_ref': 'artifact:styles'},
                    ],
                },
                bundle_root=root / 'bundles',
                created_at='2026-06-13T15:52:00Z',
            )

            self.assertEqual(payload['status'], 'failed')
            self.assertEqual(payload['link_check']['status'], 'failed')
            syntax_items = [
                item for item in payload['link_check']['missing']
                if item.get('reason') == 'syntax'
            ]
            self.assertTrue(syntax_items)
            joined_issues = '\n'.join(syntax_items[0].get('issues') or [])
            self.assertIn('malformed rel/href attributes', joined_issues)

    def test_bundle_uses_fulfilled_outputs_and_excludes_repair_intermediate_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            document_dir = root / 'artifacts' / 'documents'
            image_dir = root / 'artifacts' / 'images'
            document_dir.mkdir(parents=True)
            image_dir.mkdir(parents=True)
            index_path = document_dir / 'index.html'
            styles_path = document_dir / 'styles.css'
            image_path = image_dir / 'hero.png'
            repair_path = document_dir / 'generated-image-repair-index.html'
            index_path.write_text(
                '<!doctype html><link rel="stylesheet" href="styles.css"><img src="../images/hero.png">',
                encoding='utf-8',
            )
            styles_path.write_text('body { color: white; }', encoding='utf-8')
            image_path.write_bytes(b'png')
            repair_path.write_text('<!doctype html><p>intermediate repair file</p>', encoding='utf-8')

            payload = bundle_response_artifacts(
                {
                    'id': 'resp_repair_intermediate',
                    'lifecycle_state': 'repair_needed',
                    'outputs': [
                        {
                            'type': 'text',
                            'status': 'fulfilled',
                            'source': 'promoted_output_slot',
                            'artifact_ref': 'artifact:index',
                        },
                        {
                            'type': 'text',
                            'status': 'fulfilled',
                            'source': 'promoted_output_slot',
                            'artifact_ref': 'artifact:styles',
                        },
                        {
                            'type': 'image',
                            'status': 'fulfilled',
                            'source': 'promoted_output_slot',
                            'artifact_ref': 'artifact:hero',
                        },
                        {
                            'type': 'text',
                            'status': 'repair_needed',
                            'source': 'promoted_output_slot',
                            'artifact_ref': 'artifact:text_generated_image_bad',
                        },
                    ],
                    'artifacts': [
                        {'type': 'text', 'path': str(index_path), 'name': 'index', 'artifact_ref': 'artifact:index'},
                        {'type': 'text', 'path': str(styles_path), 'name': 'styles', 'artifact_ref': 'artifact:styles'},
                        {'type': 'image', 'path': str(image_path), 'name': 'hero', 'artifact_ref': 'artifact:hero'},
                        {
                            'type': 'text',
                            'path': str(repair_path),
                            'name': 'index',
                            'artifact_ref': 'artifact:text_generated_image_bad',
                        },
                    ],
                },
                bundle_root=root / 'bundles',
                created_at='2026-06-07T19:45:00Z',
            )

            self.assertEqual(payload['status'], 'bundled')
            copied_refs = {item['artifact_ref'] for item in payload['copied_artifacts']}
            copied_paths = {item['relative_path'] for item in payload['copied_artifacts']}
            self.assertEqual(copied_refs, {'artifact:index', 'artifact:styles', 'artifact:hero'})
            self.assertNotIn('artifact:text_generated_image_bad', copied_refs)
            self.assertNotIn('assets/files/index.html', copied_paths)
            self.assertTrue((Path(payload['bundle_path']) / 'index.html').exists())
            self.assertTrue((Path(payload['bundle_path']) / 'assets/css/styles.css').exists())
            self.assertTrue((Path(payload['bundle_path']) / 'assets/images/hero.png').exists())

    def test_bundle_includes_local_dependencies_referenced_by_public_html(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            document_dir = root / 'artifacts' / 'documents'
            image_dir = root / 'artifacts' / 'images'
            document_dir.mkdir(parents=True)
            image_dir.mkdir(parents=True)
            index_path = document_dir / 'index.html'
            styles_path = document_dir / 'styles.css'
            repair_path = document_dir / 'repair-index.html'
            exterior_path = image_dir / 'exterior.png'
            interior_path = image_dir / 'interior.png'
            detail_path = image_dir / 'detail.png'
            index_path.write_text(
                '<!doctype html>'
                '<link rel="stylesheet" href="styles.css">'
                '<section style="background-image: url(../images/exterior.png)"></section>'
                '<img src="../images/interior.png">'
                '<img src="../images/detail.png">',
                encoding='utf-8',
            )
            styles_path.write_text('body { color: #111; }', encoding='utf-8')
            repair_path.write_text('<!doctype html><p>repair scratch</p>', encoding='utf-8')
            exterior_path.write_bytes(b'exterior')
            interior_path.write_bytes(b'interior')
            detail_path.write_bytes(b'detail')

            payload = bundle_response_artifacts(
                {
                    'id': 'resp_linked_deps',
                    'outputs': [
                        {
                            'type': 'text',
                            'status': 'fulfilled',
                            'source': 'promoted_output_slot',
                            'artifact_ref': 'artifact:index',
                        },
                        {
                            'type': 'text',
                            'status': 'fulfilled',
                            'source': 'promoted_output_slot',
                            'artifact_ref': 'artifact:styles',
                        },
                        {
                            'type': 'image',
                            'status': 'fulfilled',
                            'source': 'promoted_output_slot',
                            'artifact_ref': 'artifact:image_slot_1',
                        },
                        {
                            'type': 'text',
                            'status': 'fulfilled',
                            'source': 'promoted_output_slot',
                            'artifact_ref': 'artifact:text_generated_image_bad',
                        },
                    ],
                    'artifacts': [
                        {'type': 'text', 'path': str(index_path), 'name': 'index', 'artifact_ref': 'artifact:index'},
                        {'type': 'text', 'path': str(styles_path), 'name': 'styles', 'artifact_ref': 'artifact:styles'},
                        {'type': 'image', 'path': str(detail_path), 'name': 'detail', 'artifact_ref': 'artifact:image_slot_1'},
                        {
                            'type': 'text',
                            'path': str(repair_path),
                            'name': 'index',
                            'artifact_ref': 'artifact:text_generated_image_bad',
                        },
                    ],
                },
                bundle_root=root / 'bundles',
                created_at='2026-06-07T20:35:00Z',
            )

            self.assertEqual(payload['status'], 'bundled')
            relative_paths = {item['relative_path'] for item in payload['copied_artifacts']}
            self.assertEqual(
                relative_paths,
                {
                    'index.html',
                    'assets/css/styles.css',
                    'assets/images/detail.png',
                    'assets/images/exterior.png',
                    'assets/images/interior.png',
                },
            )
            copied_refs = {item.get('artifact_ref') for item in payload['copied_artifacts']}
            self.assertNotIn('artifact:text_generated_image_bad', copied_refs)
            bundled_index = Path(payload['entrypoint']).read_text(encoding='utf-8')
            self.assertIn('assets/images/exterior.png', bundled_index)
            self.assertIn('assets/images/interior.png', bundled_index)
            self.assertIn('assets/images/detail.png', bundled_index)

    def test_bundle_does_not_promote_css_only_artifacts_to_entrypoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            document_dir = root / 'artifacts' / 'documents'
            image_dir = root / 'artifacts' / 'images'
            document_dir.mkdir(parents=True)
            image_dir.mkdir(parents=True)
            styles_path = document_dir / 'styles.css'
            image_path = image_dir / 'hero.png'
            styles_path.write_text(
                '.hero { background: url("../images/hero.png") center/cover no-repeat; }',
                encoding='utf-8',
            )
            image_path.write_bytes(b'png')

            payload = bundle_response_artifacts(
                {
                    'id': 'resp_css_without_html',
                    'outputs': [
                        {
                            'type': 'text',
                            'status': 'fulfilled',
                            'source': 'promoted_output_slot',
                            'artifact_ref': 'artifact:styles',
                        },
                        {
                            'type': 'image',
                            'status': 'fulfilled',
                            'source': 'promoted_output_slot',
                            'artifact_ref': 'artifact:hero',
                        },
                    ],
                    'artifacts': [
                        {'type': 'text', 'path': str(styles_path), 'name': 'styles', 'artifact_ref': 'artifact:styles'},
                        {'type': 'image', 'path': str(image_path), 'name': 'hero', 'artifact_ref': 'artifact:hero'},
                    ],
                },
                bundle_root=root / 'bundles',
                created_at='2026-06-15T18:15:00Z',
            )

            self.assertEqual(payload['status'], 'bundled')
            self.assertIsNone(payload.get('entrypoint'))
            self.assertIsNone(payload.get('entrypoint_relative_path'))
            self.assertTrue((Path(payload['bundle_path']) / 'assets/css/styles.css').exists())
            self.assertTrue((Path(payload['bundle_path']) / 'assets/images/hero.png').exists())

    def test_non_index_html_entrypoint_keeps_identity_and_missing_index_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            document_dir = root / 'artifacts' / 'documents'
            document_dir.mkdir(parents=True)
            configurator_path = document_dir / 'timestamp_model_configurator.html'
            configurator_path.write_text(
                '<!doctype html>'
                '<nav><a href="index.html">Atelier</a>'
                '<a href="configurator.html">Configurator</a></nav>',
                encoding='utf-8',
            )

            payload = bundle_response_artifacts(
                {
                    'id': 'resp_configurator_without_index',
                    'artifacts': [
                        {
                            'type': 'text',
                            'path': str(configurator_path),
                            'name': 'configurator',
                            'artifact_ref': 'artifact:configurator',
                        }
                    ],
                },
                bundle_root=root / 'bundles',
                created_at='2026-08-05T04:30:00Z',
            )

            bundle_dir = Path(payload['bundle_path'])
            bundled_configurator = bundle_dir / 'configurator.html'
            self.assertEqual(payload['status'], 'failed')
            self.assertEqual(payload['entrypoint'], str(bundled_configurator))
            self.assertEqual(payload['entrypoint_relative_path'], 'configurator.html')
            self.assertTrue(bundled_configurator.exists())
            self.assertFalse((bundle_dir / 'index.html').exists())
            self.assertEqual(
                {
                    item.get('target')
                    for item in payload['link_check']['missing']
                    if item.get('reason') == 'missing'
                },
                {'index.html'},
            )
            bundled_html = bundled_configurator.read_text(encoding='utf-8')
            self.assertIn('href="configurator.html"', bundled_html)
            self.assertNotIn('href="index.html">Configurator', bundled_html)

    def test_named_follow_up_bundle_carries_only_authorized_immutable_predecessor_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            document_dir = root / 'artifacts' / 'documents'
            image_dir = root / 'artifacts' / 'images'
            document_dir.mkdir(parents=True)
            image_dir.mkdir(parents=True)
            index_path = document_dir / 'stamp_index.html'
            configurator_path = document_dir / 'stamp_configurator.html'
            styles_path = document_dir / 'stamp_styles.css'
            pricing_path = document_dir / 'stamp_pricing.json'
            old_hero_path = image_dir / 'old-hero.png'
            new_image_paths = [image_dir / f'material-{index}.png' for index in range(1, 4)]
            index_path.write_text(
                '<!doctype html>'
                '<link rel="stylesheet" href="stamp_styles.css">'
                '<a href="stamp_configurator.html">Configurator</a>'
                '<img src="../images/old-hero.png">',
                encoding='utf-8',
            )
            configurator_path.write_text(
                '<!doctype html>'
                '<link rel="stylesheet" href="stamp_styles.css">'
                '<a href="stamp_index.html">Atelier</a>'
                + ''.join(
                    f'<img src="../images/{path.name}">' for path in new_image_paths
                ),
                encoding='utf-8',
            )
            styles_path.write_text('body { color: silver; }', encoding='utf-8')
            pricing_path.write_text('{"materials": []}', encoding='utf-8')
            old_hero_path.write_bytes(b'old-hero')
            for path in new_image_paths:
                path.write_bytes(path.name.encode('utf-8'))

            carried_dependencies = [
                {
                    'type': 'text',
                    'path': str(index_path),
                    'name': 'index',
                    'artifact_ref': 'artifact:old-index',
                },
                {
                    'type': 'text',
                    'path': str(pricing_path),
                    'name': 'pricing',
                    'artifact_ref': 'artifact:old-pricing',
                },
                {
                    'type': 'image',
                    'path': str(old_hero_path),
                    'name': 'old-hero',
                    'artifact_ref': 'artifact:old-hero',
                },
            ]
            for item in carried_dependencies:
                item.update(
                    {
                        'kind': item['type'],
                        'source_path': item['path'],
                        'source_response_id': 'resp-predecessor',
                        'origin': 'canonical_predecessor_bundle_dependency',
                        'carried_public_dependency': True,
                    }
                )
            current_artifacts = [
                {
                    'type': 'text',
                    'path': str(configurator_path),
                    'name': 'configurator',
                    'artifact_ref': 'artifact:configurator-current',
                },
                {
                    'type': 'text',
                    'path': str(styles_path),
                    'name': 'styles',
                    'artifact_ref': 'artifact:styles-current',
                },
                *[
                    {
                        'type': 'image',
                        'path': str(path),
                        'name': path.stem,
                        'artifact_ref': f'artifact:new-{index}',
                    }
                    for index, path in enumerate(new_image_paths, start=1)
                ],
            ]
            outputs = [
                {
                    'type': item['type'],
                    'status': 'fulfilled',
                    'artifact_ref': item['artifact_ref'],
                }
                for item in current_artifacts
            ]

            payload = bundle_response_artifacts(
                {
                    'id': 'resp-named-follow-up',
                    'outputs': outputs,
                    'artifacts': current_artifacts,
                    'response_frame': {
                        'request': {
                            'current_predecessor_context': {
                                'kind': 'ollmo.current_predecessor_context',
                                'status': 'authorized',
                                'authorization': (
                                    'canonical_same_conversation_predecessor'
                                ),
                                'promotion_mode': 'named_text_edit',
                                'source_response_id': 'resp-predecessor',
                                'matched_text_artifacts': [
                                    {'path': str(configurator_path)},
                                    {'path': str(styles_path)},
                                ],
                                'carried_public_dependencies': carried_dependencies,
                            }
                        }
                    },
                },
                bundle_root=root / 'bundles',
                created_at='2026-08-05T06:00:00Z',
            )

            bundle_dir = Path(payload['bundle_path'])
            relative_paths = {
                item['relative_path'] for item in payload['copied_artifacts']
            }
            self.assertEqual(payload['status'], 'bundled')
            self.assertEqual(payload['entrypoint_relative_path'], 'index.html')
            self.assertIn('index.html', relative_paths)
            self.assertIn('assets/files/configurator.html', relative_paths)
            self.assertIn('assets/files/pricing.json', relative_paths)
            self.assertIn('assets/css/styles.css', relative_paths)
            self.assertIn('assets/images/old-hero.png', relative_paths)
            self.assertEqual(
                len([path for path in relative_paths if path.endswith('styles.css')]),
                1,
            )
            bundled_configurator = (
                bundle_dir / 'assets' / 'files' / 'configurator.html'
            ).read_text(encoding='utf-8')
            self.assertIn('href="../../index.html"', bundled_configurator)

    def test_named_follow_up_bundle_does_not_carry_when_a_revision_target_is_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            configurator_path = root / 'configurator.html'
            styles_path = root / 'styles.css'
            pricing_path = root / 'pricing.json'
            configurator_path.write_text('<!doctype html><p>Updated</p>', encoding='utf-8')
            styles_path.write_text('body { color: silver; }', encoding='utf-8')
            pricing_path.write_text('{}', encoding='utf-8')

            payload = bundle_response_artifacts(
                {
                    'id': 'resp-partial-named-follow-up',
                    'outputs': [
                        {
                            'type': 'text',
                            'status': 'fulfilled',
                            'artifact_ref': 'artifact:configurator-current',
                        }
                    ],
                    'artifacts': [
                        {
                            'type': 'text',
                            'path': str(configurator_path),
                            'name': 'configurator',
                            'artifact_ref': 'artifact:configurator-current',
                        }
                    ],
                    'response_frame': {
                        'request': {
                            'current_predecessor_context': {
                                'kind': 'ollmo.current_predecessor_context',
                                'status': 'authorized',
                                'authorization': (
                                    'canonical_same_conversation_predecessor'
                                ),
                                'promotion_mode': 'named_text_edit',
                                'source_response_id': 'resp-predecessor',
                                'matched_text_artifacts': [
                                    {'path': str(configurator_path)},
                                    {'path': str(styles_path)},
                                ],
                                'carried_public_dependencies': [
                                    {
                                        'type': 'text',
                                        'kind': 'text',
                                        'path': str(pricing_path),
                                        'source_path': str(pricing_path),
                                        'name': 'pricing',
                                        'artifact_ref': 'artifact:old-pricing',
                                        'source_response_id': 'resp-predecessor',
                                        'origin': (
                                            'canonical_predecessor_bundle_dependency'
                                        ),
                                        'carried_public_dependency': True,
                                    }
                                ],
                            }
                        }
                    },
                },
                bundle_root=root / 'bundles',
                created_at='2026-08-05T06:05:00Z',
            )

            self.assertEqual(
                {item.get('artifact_ref') for item in payload['copied_artifacts']},
                {'artifact:configurator-current'},
            )

    def test_bundle_does_not_fallback_to_raw_artifacts_when_outputs_are_all_unusable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            document_dir = root / 'artifacts' / 'documents'
            document_dir.mkdir(parents=True)
            repair_path = document_dir / 'generated-image-repair-index.html'
            repair_path.write_text('<!doctype html><p>intermediate repair file</p>', encoding='utf-8')

            with self.assertRaises(ValueError):
                bundle_response_artifacts(
                    {
                        'id': 'resp_only_repair_intermediate',
                        'outputs': [
                            {
                                'type': 'text',
                                'status': 'repair_needed',
                                'source': 'promoted_output_slot',
                                'artifact_ref': 'artifact:text_generated_image_bad',
                            }
                        ],
                        'artifacts': [
                            {
                                'type': 'text',
                                'path': str(repair_path),
                                'name': 'index',
                                'artifact_ref': 'artifact:text_generated_image_bad',
                            }
                        ],
                    },
                    bundle_root=root / 'bundles',
                    created_at='2026-06-07T19:46:00Z',
                )

    def test_non_artifact_response_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                bundle_response_artifacts({'id': 'resp_empty', 'output_text': 'hello'}, bundle_root=Path(tmpdir) / 'bundles')


if __name__ == '__main__':
    unittest.main()
