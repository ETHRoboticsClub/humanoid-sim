"""Small real Kamino model fixtures for optional solver mechanisms."""
import warp as wp

def model_fixture(wp,body_count,device):
    from newton._src.solvers.kamino._src.core.builder import ModelBuilderKamino
    from newton._src.solvers.kamino._src.core.joints import JointActuationType,JointDoFType
    from newton._src.solvers.kamino._src.dynamics.dual import DualProblem
    from newton._src.solvers.kamino._src.kinematics.constraints import make_unilateral_constraints_info
    from newton._src.solvers.kamino._src.kinematics.jacobians import SparseSystemJacobians
    from newton._src.solvers.kamino._src.linalg.linear import LLTBlockedSolver
    builder=ModelBuilderKamino(default_world=False);world=builder.add_world(name='fixture');identity=wp.mat33f(1.,0.,0.,0.,1.,0.,0.,0.,1.)
    bodies=[builder.add_rigid_body(m_i=1.,i_I_i=identity,q_i_0=wp.transformf(0.,0.,0.,0.,0.,0.,1.),world_index=world) for _ in range(body_count)]
    for i in range(body_count-1):
        last=i==body_count-2
        builder.add_joint(act_type=JointActuationType.PASSIVE,dof_type=JointDoFType.REVOLUTE if last else JointDoFType.FIXED,bid_B=bodies[i],bid_F=bodies[i+1],B_r_Bj=wp.vec3f(0.),F_r_Fj=wp.vec3f(0.),X_Bj=identity,a_j=.1 if last else 0.,b_j=1. if last else 0.,f_j=.1 if last else 0.,world_index=world)
    model=builder.finalize(device);data=model.data();make_unilateral_constraints_info(model,data,None,None);jac=SparseSystemJacobians(model);p=DualProblem(model=model,data=data,jacobians=jac,solver=LLTBlockedSolver,sparse=False);p.data.P.fill_(1.);p.data.bound_lower.fill_(-.2);p.data.bound_upper.fill_(.2)
    return model,p

def make():
    return model_fixture(wp, 2, "cuda:0")
