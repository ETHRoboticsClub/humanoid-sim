# humanoid-sim

The ETHRC humanoid below Controls: drives, ankle mechanism and body, on Isaac Lab 3.0 /
Newton (MJWarp). ROS-free. See `AGENTS.md` for the rules and the layout.

```bash
git submodule update --init --recursive
uv sync --extra dev --extra newton   # contract and kinematics tests: uv run pytest -m "not physics"
docker buildx build --platform linux/arm64 -f docker/Dockerfile -t humanoid-sim:dev .   # Isaac Lab image
```
