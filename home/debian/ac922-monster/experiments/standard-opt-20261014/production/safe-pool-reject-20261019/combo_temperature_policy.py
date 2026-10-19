"""Pure admission rules for actual in-band samples; no invented BMC fields."""
import math

GPU_STOP_C = 65
OTHER_STOP_C = 58

IPMI_KEYS = frozenset(('fan0_0', 'fan2_0', 'fan3_0', 'ambient', 'pcie'))


def fitted_inventory(sample):
    values = sample['hwmon_readings']
    labels = frozenset(k for k, v in values.items() if ' DIMM ' not in k or float(v) > 0)
    required = {'Chip 0 DIMM 5', 'Chip 8 DIMM 5', 'Chip 0 VRM VDD', 'Chip 8 VRM VDD'}
    if not required <= labels:
        raise ValueError('missing fitted DIMM/VDD inventory')
    if sum(' GPU ' in k and not k.endswith(' MEM') for k in labels) != 6:
        raise ValueError('require all six GPU cores')
    if sum(' GPU ' in k and k.endswith(' MEM') for k in labels) != 6:
        raise ValueError('require all six GPU memory sensors')
    if sum(' Core ' in k for k in labels) < 40:
        raise ValueError('incomplete CPU inventory')
    return labels


def inspect(sample, expected_labels, host_now):
    errors = []
    low_fans = []
    if sample.get('errors'):
        errors.append('collector_errors')
    start = sample.get('sample_started_unix', -1)
    end = sample.get('sample_completed_unix', -1)
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return ['invalid_timestamp'], []
    if not all(math.isfinite(x) for x in (start, end, host_now)) or not (0 <= end-start <= 10) or not (-1 <= host_now-end <= 15):
        errors.append('stale_or_slow_snapshot')
    ipmi = sample.get('ipmi_readings', {})
    hwmon = sample.get('hwmon_readings', {})
    if not IPMI_KEYS <= ipmi.keys() or not expected_labels <= hwmon.keys():
        return errors + ['missing_sensor'], []
    for key in expected_labels:
        value = hwmon[key]
        limit = GPU_STOP_C if " GPU " in key else OTHER_STOP_C
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not (5 <= value < limit):
            errors.append('temperature:' + key)
    for key in ('ambient', 'pcie'):
        value = ipmi[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not (5 <= value < OTHER_STOP_C):
            errors.append('temperature:' + key)
    for key in ('fan0_0', 'fan2_0', 'fan3_0'):
        value = ipmi[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not (0 <= value <= 12000):
            errors.append('invalid_tach:' + key)
        elif value < 2700:
            low_fans.append(key)
    return errors, low_fans
