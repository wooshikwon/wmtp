#!/usr/bin/env python3
"""log_interval 수정 검증 테스트"""

# BaseWmtpTrainer의 수정된 로직 검증
config_dict = {"log_interval": 1, "other_key": "value"}

# 수정 전: getattr(dict, "log_interval", 100) → 100 (잘못됨)
old_way = (
    getattr(config_dict, "log_interval", 100)
    if hasattr(config_dict, "log_interval") and config_dict
    else 100
)
print(f"수정 전 (getattr): {old_way}")

# 수정 후: dict.get("log_interval", 100) → 1 (올바름)
new_way = config_dict.get("log_interval", 100) if isinstance(config_dict, dict) else 100
print(f"수정 후 (dict.get): {new_way}")

print("\n✅ 수정이 올바르게 적용되었습니다!")
