"""Execute only the real selector functions; no credentials or network calls."""
import ast
import ipaddress
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

SOURCE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/home/debian/apigateway/aliyun_ddns.py')
sys.argv = sys.argv[:1]
NEW='240e:3bb:2eb4:670:72e2:84ff:fe14:474f'
OLD='240e:3bb:2eb4:31e0:72e2:84ff:fe14:474f'


def select(lines, route):
    source = ast.parse(SOURCE.read_text())
    functions = [n for n in source.body if isinstance(n,ast.FunctionDef)
                 and n.name in {'current_ipv6','select_current_ipv6'}]
    def run(argv, **kwargs):
        return SimpleNamespace(stdout=lines if 'addr' in argv else route, returncode=0)
    scope = {'subprocess':SimpleNamespace(run=run),'IFACE':'enP5p1s0f0','ipaddress':ipaddress}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(SOURCE),'exec'),scope)
    return scope['current_ipv6']()


def addresses(*values):
    return '\n'.join(f'    inet6 {a}/64 scope global dynamic mngtmpaddr' for a in values)


class IPv6SourceTests(unittest.TestCase):
    def test_active_route_beats_old_last(self):
        self.assertEqual(select(addresses(NEW,OLD),'2400:3200::1 dev enP5p1s0f0 src '+NEW),NEW)
    def test_route_is_used_not_first_address(self):
        self.assertEqual(select(addresses(NEW,OLD),'2400:3200::1 dev enP5p1s0f0 src '+OLD),OLD)
    def test_single_fallback(self):
        self.assertEqual(select(addresses(NEW),''),NEW)
    def test_ambiguous_route_failure(self):
        self.assertIsNone(select(addresses(NEW,OLD),''))
    def test_temporary_source_maps_to_stable_prefix(self):
        self.assertEqual(select(addresses(NEW,OLD),'2400:3200::1 src 240e:3bb:2eb4:670:1234:5678:1111:2222'),NEW)
    def test_temporary_and_deprecated_excluded(self):
        text=addresses(NEW)+'\ninet6 '+OLD+'/64 scope global deprecated\ninet6 240e:3bb::42/64 scope global temporary'
        self.assertEqual(select(text,'2400:3200::1 src '+NEW),NEW)
    def test_tentative_and_failed_excluded(self):
        text=addresses(NEW)+'\ninet6 '+OLD+'/64 scope global tentative dadfailed'
        self.assertEqual(select(text,'2400:3200::1 src '+NEW),NEW)
    def test_private_address_not_published(self):
        self.assertIsNone(select(addresses('fd7a:115c:a1e0::42'),''))
    def test_no_address(self):
        self.assertIsNone(select('',''))


if __name__=='__main__':
    unittest.main()
