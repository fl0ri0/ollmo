import datetime as dt
import re
import unittest
from pathlib import Path

from ollmo_core.transports import _timestamp_prefix
from ollmo_services.events import make_event
from ollmo_services.settings_artifacts import build_settings_artifact


REPO_ROOT = Path(__file__).resolve().parent.parent
ACTIVE_SOURCE_ROOTS = (
    REPO_ROOT / 'helpers',
    REPO_ROOT / 'ollmo_core',
    REPO_ROOT / 'ollmo_g',
    REPO_ROOT / 'ollmo_integrations',
    REPO_ROOT / 'ollmo_orchestration',
    REPO_ROOT / 'ollmo_runtime',
    REPO_ROOT / 'ollmo_server',
    REPO_ROOT / 'ollmo_services',
    REPO_ROOT / 'scripts',
    REPO_ROOT / 'state' / 'ollmo_run_monitor',
    REPO_ROOT / 'tests',
)


def _active_python_sources() -> list[Path]:
    sources = sorted(REPO_ROOT.glob('*.py'))
    for source_root in ACTIVE_SOURCE_ROOTS:
        if source_root.exists():
            sources.extend(sorted(source_root.rglob('*.py')))
    return sources


def _parse_canonical_utc(value: str) -> dt.datetime:
    if not value.endswith('Z'):
        raise AssertionError(f'UTC timestamp must end with Z: {value!r}')
    parsed = dt.datetime.fromisoformat(value[:-1] + '+00:00')
    if parsed.utcoffset() != dt.timedelta(0):
        raise AssertionError(f'UTC timestamp must carry a zero offset: {value!r}')
    return parsed


class TimezoneAwareUtcTests(unittest.TestCase):
    def test_active_python_sources_do_not_call_deprecated_utcnow(self):
        pattern = re.compile(r'\.utcnow\s*\(')
        offenders: list[str] = []

        for path in _active_python_sources():
            for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
                if pattern.search(line):
                    offenders.append(f'{path.relative_to(REPO_ROOT)}:{line_number}: {line.strip()}')

        self.assertEqual(offenders, [], 'deprecated utcnow() calls remain:\n' + '\n'.join(offenders))

    def test_iso_timestamp_producers_keep_canonical_utc_wire_format(self):
        event = make_event(category='test', action='timestamp', status='ok')
        settings = build_settings_artifact(
            {
                'kind': 'ollmo.control_snapshot',
                'values': {'generation': {'temperature': 0.2}},
            }
        )

        _parse_canonical_utc(event['timestamp'])
        _parse_canonical_utc(settings['created_at'])

    def test_compact_transport_timestamp_keeps_utc_token_shape(self):
        self.assertRegex(_timestamp_prefix(), r'^\d{8}T\d{6}Z$')


if __name__ == '__main__':
    unittest.main()
