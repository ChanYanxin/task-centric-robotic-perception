from tc_perception.synthetic import generate
from tc_perception.benchmark import benchmark


def test_reproducible_benchmark(tmp_path):
    video,gt=generate(tmp_path)
    report=benchmark(str(video),str(tmp_path/"results"),str(gt),max_frames=50)
    modes=report["results"]
    assert [row["strategy"] for row in modes]==["continuous","roi","event"]
    assert all(row["frames"]==50 for row in modes)
    assert modes[2]["frames_offloaded"] < modes[0]["frames_offloaded"]
    assert modes[2]["sent_bytes"] < modes[0]["sent_bytes"]
    assert (tmp_path/"results"/"benchmark_summary.csv").exists()
