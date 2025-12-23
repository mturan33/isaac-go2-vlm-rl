"""
G1 Standalone Test - DDS gerektirmez, sadece simülasyon
"""

import argparse
import os

# PROJECT_ROOT'u ayarla
os.environ["PROJECT_ROOT"] = r"C:\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\unitree_sim_isaaclab"

from isaacsim import SimulationApp

# Simulation başlat
simulation_app = SimulationApp({"headless": False})

# Isaac imports (SimulationApp'ten sonra!)
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

# G1 config'i import et
from external.unitree_robots.unitree import G129_CFG_WITH_DEX1_BASE_FIX


def main():
    """G1 robot'u basitçe yükle ve göster."""

    # Simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    sim = SimulationContext(sim_cfg)

    # Ground plane ekle
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    # Light ekle
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # G1 Robot'u spawn et
    robot_cfg = G129_CFG_WITH_DEX1_BASE_FIX.copy()
    robot_cfg.prim_path = "/World/G1"
    robot = Articulation(cfg=robot_cfg)

    # Simulation'ı başlat
    sim.reset()

    print("\n" + "=" * 50)
    print("G1 Robot başarıyla yüklendi!")
    print(f"Joint sayısı: {robot.num_joints}")
    print(f"Body sayısı: {robot.num_bodies}")
    print("=" * 50 + "\n")

    # Ana döngü
    count = 0
    while simulation_app.is_running():
        # Her 500 step'te bir reset
        if count % 500 == 0:
            robot.reset()
            print(f"[Step {count}] Robot reset edildi")

        # Simülasyon adımı
        sim.step()
        robot.update(sim.get_physics_dt())
        count += 1

    # Cleanup
    simulation_app.close()


if __name__ == "__main__":
    main()