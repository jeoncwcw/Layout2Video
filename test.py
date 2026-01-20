import tarfile
from pathlib import Path

def check_shards(wds_path):
    for shard in Path(wds_path).rglob("*.tar"):
        try:
            with tarfile.open(shard, "r") as t:
                # 파일 목록을 끝까지 읽어보며 손상 여부 확인
                t.getmembers()
        except Exception as e:
            print(f"❌ Corrupted Shard Found: {shard}")
            print(f"   Reason: {e}")

# 창우님의 wds_root 경로를 넣어주세요
check_shards("./datasets/betr_wds")