from mujoco import MjModel, MjData, mj_forward, mj_collision
import numpy as np

model = MjModel.from_xml_path("model/1018model.xml")


def min_contact_dist(data):
    mj_forward(model, data)
    mj_collision(model, data)

    if data.ncon == 0:
        return None

    return min(float(data.contact[i].dist) for i in range(data.ncon))


base_qpos = model.key_qpos[0].copy()

for idx in [0, 1]:
    print()
    print("========== qpos index", idx, "==========")

    for delta in np.linspace(-0.05, 0.05, 11):
        data = MjData(model)

        data.qpos[:] = base_qpos.copy()
        data.qpos[idx] += delta
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0

        d = min_contact_dist(data)

        print(
            "delta=",
            round(float(delta), 4),
            "qpos=",
            data.qpos[idx],
            "ncon=",
            data.ncon,
            "min_dist=",
            d,
        )

for i in range(data.ncon):
    contact = data.contact[i]

    geom1_id = contact.geom1
    geom2_id = contact.geom2

    geom1_name = model.geom(geom1_id).name
    geom2_name = model.geom(geom2_id).name

    print(
        i,
        "geom1_id=", geom1_id,
        "geom1_name=", geom1_name,
        "geom2_id=", geom2_id,
        "geom2_name=", geom2_name,
        "dist=", contact.dist,
        "pos=", contact.pos,
    )