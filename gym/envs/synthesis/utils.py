import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
from scipy.interpolate import BSpline, splprep, splev
import scipy.integrate as integrate
import time
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
import pickle
from scipy.interpolate import UnivariateSpline
from collections import deque

def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

def triangulate_point(l1, l2, l3):
    if l1*l2 > 1e-8:
        theta = np.arccos((l3**2 - (l1**2 + l2**2))/(2*l1*l2))
    else:
        theta = 0
    return theta

def retrive_curves_from_features(fvs, n_points=50):
    assert(len(fvs.shape) == 3)
    n_samples, n_5, _ = fvs.shape

    curves = np.zeros((n_samples, n_points, 2))

    for ind, fv in enumerate(fvs):
        f_mat = np.zeros((n_points,n_points))
        k = 0
        for i in range(n_points):
            for j in range(n_points):
                if i < j and i + 3 >= j:
                    f_mat[i][j] = fv[k,0]
                    k += 1

        x = [0, f_mat[0][1]]
        y = [0, 0]
        theta0 = 0.0

        for i in range(2, n_points):
            theta = triangulate_point(f_mat[i-2][i-1], f_mat[i-1][i], f_mat[i-2][i])
            if i != 2:
                xx1 = x[i-1] + f_mat[i-1][i]*np.cos(theta0 - theta)
                yy1 = y[i-1] + f_mat[i-1][i]*np.sin(theta0 - theta)

                xx2 = x[i-1] + f_mat[i-1][i]*np.cos(theta + theta0)
                yy2 = y[i-1] + f_mat[i-1][i]*np.sin(theta + theta0)

                dist1 = np.abs(f_mat[i-3][i]-np.sqrt((xx1-x[i-3])**2 + (yy1-y[i-3])**2))
                dist2 = np.abs(f_mat[i-3][i]-np.sqrt((xx2-x[i-3])**2 + (yy2-y[i-3])**2))
                if dist1 <= dist2 :
                    theta = -theta

            xx = f_mat[i-1][i]*np.cos(theta + theta0)
            yy = f_mat[i-1][i]*np.sin(theta + theta0)
            theta0 = theta + theta0

            x.append(xx + x[i-1])
            y.append(yy + y[i-1])

        curves[ind, :, 0] = np.array(x)
        curves[ind, :, 1] = np.array(y)

    return curves

def features_from_curves(coupler_motion, n_points=50):
    feature_vector_cm = []

    n_points = 50

    i = 0
    j = 0
    cm = np.zeros((coupler_motion.shape[0],n_points,3))
    step = int((coupler_motion.shape[1])/(n_points-1))
    for _ in range(n_points):
        if i == 0:
            cm[:,j,:] = np.copy(coupler_motion[:,i,:])
        else:
            cm[:,j,:] = np.copy(coupler_motion[:,i-1,:])
        i += step
        j += 1

    k = 0
    for i in range(n_points):
        for j in range(n_points):
            if i < j and j <= i + 3 :
                a = np.sqrt((cm[:,j,0] - cm[:,i,0])**2 + (cm[:,j,1] - cm[:,i,1])**2)
                feature_vector_cm.append(a)
                k += 1

    feature_vector_cm = np.array([feature_vector_cm])
    feature_vector_cm = np.transpose(feature_vector_cm, [2,1,0])
    return feature_vector_cm

def get_transformed_joints_data(d):
    coupler_motion = []
    arc1_path = []
    arc2_path = []
    center_point1 = []
    center_point2 = []
    sixbar_2r_fg = []
    sixbar_2r_arc = []
    sixbar_fourbar_motion = []

    for i, (curve, joints, sign) in enumerate(zip(d['coupler_curves'].curves, d['coupler_curves'].joints, d['coupler_curves'].signs), 0):
        if len(curve) != 0:
            try:
                x = np.copy(np.array([pt['Cp0'][0] for pt in joints]))
                y = np.copy(np.array([pt['Cp0'][1] for pt in joints]))
                angle = np.copy(np.array([pt['Cp0'][2] for pt in joints]))
            except:
                x = np.copy(np.array([pt['C'][0] for pt in joints]))
                y = np.copy(np.array([pt['C'][1] for pt in joints]))
                angle = np.copy(np.array([pt['theta'] for pt in joints]))

            x, y, angle = parametrize_path(x, y, angle)
            angle = normalize_angle(angle)

            x, y, scale, dx, dy = normalize_path(x, y, return_tf=True)
            phi = -get_pca_inclination(x, y)


            cmx = np.array([pt['C'][0] for pt in joints])
            cmy = np.array([pt['C'][1] for pt in joints])

            cmx, cmy = parametrize_path(cmx, cmy)
            cm = np.array([[cmx, cmy, np.ones(cmy.shape)]])
            cm = np.transpose(cm, [0,2,1])


            b0x = np.array([pt['B0'][0] for pt in joints])
            b0y = np.array([pt['B0'][1] for pt in joints])

            b1x = np.array([pt['B1'][0] for pt in joints])
            b1y = np.array([pt['B1'][1] for pt in joints])

            try:
                sixcp1x = np.array([pt['Cp0'][0] for pt in joints])
                sixcp1y = np.array([pt['Cp0'][1] for pt in joints])

                e0x = np.array([pt['E0'][0] for pt in joints])
                e0y = np.array([pt['E0'][1] for pt in joints])

                d0x = joints[0]['D0'][0]
                d0y = joints[0]['D0'][1]

            except KeyError:
                pass


            b0x, b0y = parametrize_path(b0x, b0y)
            b1x, b1y = parametrize_path(b1x, b1y)


            try:
                sixcp1x, sixcp1y = parametrize_path(sixcp1x, sixcp1y)
                e0x, e0y = parametrize_path(e0x, e0y)
            except:
                pass

            arc1 = np.transpose([[b0x, b0y, np.ones(b0x.shape)]], [0,2,1] )
            arc2 = np.transpose([[b1x, b1y, np.ones(b0x.shape)]], [0,2,1] )

            try:
                sixcp = np.transpose([[sixcp1x, sixcp1y, np.ones(b0x.shape)]], [0,2,1] )
                sixarc = np.transpose([[e0x, e0y, np.ones(b0x.shape)]], [0,2,1] )
                sixfg = np.array([[[d0x, d0y, 1.0]]])
            except:
                pass

            cp = np.array([[[0.0, 0.0, 1.0]]])
            cp2 = np.array([[[1.0, 0.0, 1.0]]])

            cm = planar_rigid_body_transformation_on_path(cm, t=[-dx, -dy], phi=0, Trans=None)
            cm = planar_rigid_body_transformation_on_path(cm, t=[0, 0], phi=phi, Trans=None)

            arc1 = planar_rigid_body_transformation_on_path(arc1, t=[-dx, -dy], phi=0, Trans=None)
            arc1 = planar_rigid_body_transformation_on_path(arc1, t=[0, 0], phi=phi, Trans=None)

            arc2 = planar_rigid_body_transformation_on_path(arc2, t=[-dx, -dy], phi=0, Trans=None)
            arc2 = planar_rigid_body_transformation_on_path(arc2, t=[0, 0], phi=phi, Trans=None)

            try:
                sixarc = planar_rigid_body_transformation_on_path(sixarc, t=[-dx, -dy], phi=0, Trans=None)
                sixarc = planar_rigid_body_transformation_on_path(sixarc, t=[0, 0], phi=phi, Trans=None)

                sixfg = planar_rigid_body_transformation_on_path(sixfg, t=[-dx, -dy], phi=0, Trans=None)
                sixfg = planar_rigid_body_transformation_on_path(sixfg, t=[0, 0], phi=phi, Trans=None)

                sixcp = planar_rigid_body_transformation_on_path(sixcp, t=[-dx, -dy], phi=0, Trans=None)
                sixcp = planar_rigid_body_transformation_on_path(sixcp, t=[0, 0], phi=phi, Trans=None)
            except:
                pass

            cp = planar_rigid_body_transformation_on_path(cp, t=[-dx, -dy], phi=0, Trans=None)
            cp = planar_rigid_body_transformation_on_path(cp, t=[0, 0], phi=phi, Trans=None)

            cp2 = planar_rigid_body_transformation_on_path(cp2, t=[-dx, -dy], phi=0, Trans=None)
            cp2 = planar_rigid_body_transformation_on_path(cp2, t=[0, 0], phi=phi, Trans=None)

            arc1 = planar_scaling_on_path(arc1, 1.0/scale)
            arc2 = planar_scaling_on_path(arc2, 1.0/scale)
            cm = planar_scaling_on_path(cm, 1.0/scale)
            cm[:,:,2] = angle

            try:
                sixcp = planar_scaling_on_path(sixcp, 1.0/scale)
                sixarc = planar_scaling_on_path(sixarc, 1.0/scale)
                sixfg = planar_scaling_on_path(sixfg, 1.0/scale)
            except:
                pass

            cp = planar_scaling_on_path(cp, 1.0/scale)
            cp2 = planar_scaling_on_path(cp2, 1.0/scale)

            try:
                coupler_motion.append(sixcp)
                sixbar_2r_arc.append(sixarc)
                sixbar_2r_fg.append(sixfg)
                sixbar_fourbar_motion.append(cm)
            except:
                coupler_motion.append(cm)
            arc1_path.append(arc1[:,:,:2])
            arc2_path.append(arc2[:,:,:2])
            center_point1.append(cp[:,:,:2])
            center_point2.append(cp2[:,:,:2])

    coupler_motion = np.reshape(coupler_motion, [-1, 100, 3])
    arc1_path = np.reshape(arc1_path, [-1, 100, 2])
    arc2_path = np.reshape(arc2_path, [-1, 100, 2])
    arc2_path = np.reshape(arc2_path, [-1, 100, 2])
    center_point1 = np.reshape(center_point1, [-1, 1, 2])
    center_point2 = np.reshape(center_point2, [-1, 1, 2])
    if len(sixbar_2r_arc) != 0:
        sixbar_2r_arc = np.reshape(sixbar_2r_arc, [-1, 100, 2])
        sixbar_2r_fg = np.reshape(sixbar_2r_fg, [-1, 1, 2])
        sixbar_fourbar_motion = np.reshape(sixbar_fourbar_motion, [-1, 100, 3])

    return coupler_motion, arc1_path, arc2_path, center_point1, center_point2, sixbar_2r_arc, sixbar_2r_fg, sixbar_fourbar_motion

def normalize_angle(angle):
    diff_angle = [0]
    for i in range(1,len(angle)):
            diff_angle.append(angle[i] - angle[i-1])
            if np.abs(diff_angle[i]) >= np.pi:
                # Calculate absolute diff
                delta = 2*np.pi - np.abs(diff_angle[i])
                diff_angle[i] = - delta*np.sign(diff_angle[i])

    # Normalizing angle
    angle = np.cumsum(diff_angle)
    return np.copy(angle)

def signature(x, y, angle=None, steps=100, output_steps = 50):
    """ Obtains the invariant signature of motion
    """
    u = np.arange(0, 1, 1/steps)

    if angle is not None:
        diff_angle = [0]
        for i in range(1,len(angle)):
                diff_angle.append(angle[i] - angle[i-1])
                if np.abs(diff_angle[i]) >= np.pi:
                    # Calculate absolute diff
                    delta = 2*np.pi - np.abs(diff_angle[i])
                    diff_angle[i] = - delta*np.sign(diff_angle[i])

        # Normalizing angle
        angle = np.cumsum(diff_angle)
        normalized_angle = np.copy(angle)
        tck_angle, _ = splprep(x=[angle], k=3, s=0)
        angle = splev(u, tck_angle)[0]

    # Step 1
    """ 1. This step covers any pre-processing required to de-noise and convert
            the input curve to a parametric representation such that the curvature
            can be readily and robustly estimated for the curves. For example if
            the input is a digitized point set, one option is to approximate with a
            parametric cubic B-spline curve.
    """
    # Preparing Spline of fitting Path
    tck, _ = splprep(x=[x, y], k=3, s=0)
    # Evaluating Spline for many points for smoother evaluation
    x, y = splev(u, tck)

    """
        2. Sample the curvature of the curve at equal arc length intervals. Figure
             shows curvature K w.r.t. the arc length plot.
    """

    # Velocity
    v = splev(u, tck, der=1)
    # Accelation
    a = splev(u, tck, der=2)
    # Cross Product
    cur_z = - a[0]*v[1] + a[1]*v[0]
    # Curvature
    curvature = cur_z / np.power((v[0]**2 + v[1]**2), 1)
    """
        3. Integrate the unsigned curvatures along the curve by summing the absolute
            values of curvatures discretely sampled along the curve. Plot the
            integral of the unsigned curvature K =  |K| w.r.t. the arc length s ;
            see Figure 4(c).
    """
    # Cumulative Integral Calculation
    K = integrate.cumtrapz(np.abs(curvature), u)
    K = np.concatenate((np.array([0]), K))

    """
        4. Compute curvature (k) of the curve at equal-interval-sampled points
            along the integral of unsigned curvature axis (vertical axis in previous
            figure 4(c)) that is, (signed) curvature (k) w.r.t. the integral of unsigned
            curvature (K) plot see Figure 4(d). This is our signature and the core
            of our method. This can also be considered as a novel scale invariant
            parameterization of a curve.
    """
    # Fitting splines through curvature plots
    # Uniform Arc Length Parametrization
    u_cur = [0]
    dist = 0
    for i in range(1,len(K)):
        dist = dist + np.power((K[i] - K[i-1])**2, 0.5)
        u_cur.append(dist)
    u_cur = u_cur/u_cur[-1]

    # Preparing Spline of fitting Path
    #tck_cur, _ = splprep(x=[K, curvature, angle, u], u=u_cur, k=3, s=0)
    tck_K, _ = splprep(x=[K], u=u_cur, k=3, s=0)
    tck_cur, _ = splprep(x=[curvature], u=u_cur, k=3, s=0)
    if angle is not None:
        tck_ang, _ = splprep(x=[angle], u=u_cur, k=3, s=0)
    # Evaluating Spline for many points for smoother evaluation
    # adaptive step size based on curvature integral
    # for comparing correlation steps = 1/(K[-1]*100)
    u_ = np.arange(0, 1, 1.0/output_steps)
    K_init, curvature_init = K, curvature
    #K, curvature, angle, _ = splev(u_, tck_cur)
    K = splev(u_, tck_K)[0]
    curvature = splev(u_, tck_cur)[0]
    if angle is not None:
        angle = splev(u_, tck_ang)[0]

    u_1 = np.arange(0, 1, 1.0/K[-1]/100)
    #_, path_sign, motion_sign, u_new = splev(u_1, tck_cur)
    path_sign = splev(u_1, tck_cur)[0]
    if angle is not None:
        motion_sign = splev(u_1, tck_ang)[0]

    if angle is not None:
        return {'path_sign': path_sign, 'motion_sign':motion_sign, 'fixed_path_sign':np.array([curvature, K]), 'fixed_motion_sign':np.array([angle, K]), 'normalized_angle':normalized_angle, 'x':x, 'y':y}
    else:
        return {'path_sign': path_sign, 'fixed_path_sign':np.array([curvature, K]), 'x':x, 'y':y}

class CouplerCurves:
    def __init__(self):
        self.curv1 = []
        self.curv2 = []
        self.curv3 = []
        self.curv4 = []
        self.curv5 = []
        self.curv6 = []
        self.curv7 = []
        self.curv8 = []
        self.circuit = True
        self.signs = []
        self.curve_type = []
        self.crank_changed = False
        self.joints0 = []
        self.joints1 = []
        self.joints2 = []
        self.joints3 = []
        self.joints4 = []
        self.joints5 = []
        self.joints6 = []
        self.joints7 = []

    def push_point(self, points):
        if self.crank_changed:
            if self.circuit:
                self.curv5.append(points[0])
                self.curv6.append(points[1])
            else:
                self.curv7.append(points[0])
                self.curv8.append(points[1])
        else:
            if self.circuit:
                self.curv1.append(points[0])
                self.curv2.append(points[1])
            else:
                self.curv3.append(points[0])
                self.curv4.append(points[1])

    def push_joints(self, points):
        if self.crank_changed:
            if self.circuit:
                self.joints4.append(points[0])
                self.joints5.append(points[1])
            else:
                self.joints6.append(points[0])
                self.joints7.append(points[1])
        else:
            if self.circuit:
                self.joints0.append(points[0])
                self.joints1.append(points[1])
            else:
                self.joints2.append(points[0])
                self.joints3.append(points[1])

    def change_crank(self):
        self.circuit = True
        self.crank_changed = True

    def change_circuit(self):
        if self.crank_changed:
            if len(self.curv5) != 0:
                self.circuit = not self.circuit
        else:
            if len(self.curv1) != 0:
                self.circuit = not self.circuit

    def finish(self, only_first_curve=False):
        self.curv1 = np.array(self.curv1)
        self.curv2 = np.array(self.curv2)
        self.curv3 = np.array(self.curv3)
        self.curv4 = np.array(self.curv4)
        self.curv5 = np.array(self.curv5)
        self.curv6 = np.array(self.curv6)
        self.curv7 = np.array(self.curv7)
        self.curv8 = np.array(self.curv8)
        self.curves = []
        self.joints = []
        self.curves_all = [self.curv1, self.curv2, self.curv3, self.curv4, self.curv5, self.curv6, self.curv7, self.curv8]
        full_joint_data = [self.joints0, self.joints1, self.joints2, self.joints3, self.joints4, self.joints5, self.joints6, self.joints7]
        if only_first_curve:
            if len(self.curv1) >= 4:
                if len(curve) == 360:
                    self.curve_type.append('grashof')
                else:
                    self.curve_type.append('non-grashof')
                self.signs.append(signature(x=self.curv1[:,0], y=self.curv1[:,1], angle=self.curv1[:,2]))
                self.curves.append(np.array(self.curv1))
        else:
            for curve, joints in zip(self.curves_all, full_joint_data):
                if len(curve) >= 4:
                    if len(curve) == 360:
                        self.curve_type.append('grashof')
                    else:
                        self.curve_type.append('non-grashof')
                    self.signs.append(signature(x=curve[:,0], y=curve[:,1], angle=curve[:,2]))
                    self.curves.append(np.array(curve))
                    self.joints.append(joints)

    def plot_curves(self, ax, label='', mark='-*'):
        i = 0
        if len(self.curv1) != 0:
            ax.plot(self.curv1[:,0], self.curv1[:,1], mark , ms=1,lw=1, label=label+'%d'%i)
            i += 1
            ax.plot(self.curv2[:,0], self.curv2[:,1], mark , ms=1,lw=1, label=label+'%d'%i)
            i += 1
        if len(self.curv3) != 0:
            ax.plot(self.curv3[:,0], self.curv3[:,1], mark , ms=1,lw=1, label=label+'%d'%i)
            i += 1
            ax.plot(self.curv4[:,0], self.curv4[:,1], mark , ms=1,lw=1, label=label+'%d'%i)
            i += 1
        if len(self.curv5) != 0:
            ax.plot(self.curv5[:,0], self.curv5[:,1], mark , ms=1,lw=1, label=label+'%d'%i)
            i += 1
            ax.plot(self.curv6[:,0], self.curv6[:,1], mark , ms=1,lw=1, label=label+'%d'%i)
            i += 1
        if len(self.curv7) != 0:
            ax.plot(self.curv7[:,0], self.curv7[:,1], mark , ms=1,lw=1, label=label+'%d'%i)
            i += 1
            ax.plot(self.curv8[:,0], self.curv8[:,1], mark , ms=1,lw=1, label=label+'%d'%i)
        return ax

def normalize_path(x, y, return_tf=False):
    var = np.sqrt(np.var(x) + np.var(y))
    xx = np.copy((x - np.mean(x))/var)
    yy = np.copy((y - np.mean(y))/var)
    if return_tf:
        return xx, yy, var, np.mean(x), np.mean(y)
    else:
        return xx, yy

def parametrize_path(x, y, angle=None,steps=100):
    u = np.arange(0, 1, 1/steps)
    # Evaluating Spline for many points for smoother evaluation
    if angle is None:
        tck, _ = splprep(x=[x, y], k=3, s=0)
        xx, yy = splev(u, tck)
        return xx, yy
    else:
        #angle = np.array([angle])
        tck, _ = splprep(x=[x, y], k=3, s=0)
        xx, yy = splev(u, tck)

        lin_sp = np.linspace(0, 1, angle.shape[0])
        ang = UnivariateSpline(lin_sp, angle, s=0)
        angle_ = ang(np.linspace(0, 1, steps))
        return xx, yy, angle_

def sample_logNormal_link_params(random_params=True):
    params = np.zeros((5,))
    success = False
    while not success:
        for i in range(3):
            redo = True
            while redo:
                a = np.random.lognormal(0, 2, 1)
                # making sure that we get ratio less than 5 and greater than 0.2
                if a < 5 and a > 0.2:
                    redo = False
                    params[i] = a
        for i in range(3, 5):
            params[i] = np.random.normal(0, 3, 1)
        if not random_params:
            params = np.array([1.35803273, 3.58919406, 3.44406587, 0.29715139, 0.80377864])
        success = is_feasible_fourbar(params)
    return params

def is_feasible_fourbar(params):
    if np.min(params[:3]) < 0.2 or np.max(params[:3]) > 5.0:
        return False
    links = np.array([1, params[0], params[1], params[2]])
    lsorted = np.sort(links)
    if lsorted[-1]*1.4 > np.sum(lsorted[:-1]):
        return False
    else:
        return True

def simulate_fourbar(params, start = None, both_branches=True, all_joints=True, fb_type='fourR', kwargs=None):
    l1,l2,l3,l4,l5 = params
    circuit_changed = False
    if start is None:
        start = np.random.uniform(0, 2*np.pi)
    timing = np.arange(0.0 + start, 2*np.pi + start, np.pi/180.0)
    temp = deque()
    joint_data = deque()
    i = 0
    incr = 1
    circuit_break = False
    while not (circuit_break or i == 360):
        if fb_type=='fourR':
            success, output = fourbar_fk(l1,l2,l3,l4,l5,timing[i], all_joints=True, kwargs=kwargs)
        else:
            success, output = slidercrank_fk(l1,l2,l3,l4,l5,timing[i], all_joints=True, kwargs=kwargs)
        if success:
            if incr == 1:
                temp.append(output[0])
            else:
                temp.appendleft(output[0])
        else:
            if not circuit_break and len(temp) > 0:
                joint_data = temp.copy()
                temp = deque()
                circuit_break = True
                i = 0
                incr = -1
            elif circuit_break:
                break
        i += incr
    joint_data = temp + joint_data
    return joint_data

def fourbar_fk(l1,l2,l3,l4,l5,theta, all_joints=False, kwargs=None):
    """ Calculates forward kinematics
        returns a dict of joint information and coupler_angles
    """
    if kwargs is not None:
        try:
            fg = kwargs['fg']
            sg = kwargs['sg']
        except KeyError:
            fg = np.array([0, 0])
            sg = np.array([1, 0])
    else:
        fg = np.array([0, 0])
        sg = np.array([1, 0])
    # fg = Crank ground joint
    # sg = Second ground joind
    # First Floating Point
    fe = fg +  np.array([l1*np.cos(theta), l1*np.sin(theta)])
    condition = (l2 + l3 > getDistance(sg,fe)) and (abs(l2 - l3) < getDistance(sg,fe))
    if not condition:
        return False, [0]

    c1, c2, l3_inclination1, l3_inclination2, se1, se2 = fk_interm_step(fe, sg, l2, l3, l4, l5)

    if all_joints:
        return True, [{'A0': fg, 'A1': sg, 'B0': fe, 'B1': se1, 'phi': theta, 'C':c1, 'theta': l3_inclination1},{'A0': fg, 'A1': sg, 'B0': fe, 'B1': se2, 'phi': theta, 'C':c2, 'theta': l3_inclination2}]
    else:
        return True, np.array([[c1[0], c1[1], l3_inclination1], [c2[0], c2[1], l3_inclination2]])

def slidercrank_fk(l1,l2,l3,l4,phi,theta, all_joints=False, **kwargs):
    """ Calculates forward kinematics
        returns a dict of joint information and coupler_angles
    """
    if kwargs is not None:
        try:
            fg = kwargs['fg']
            sg = kwargs['sg']
        except KeyError:
            fg = np.array([np.cos(phi), np.sin(phi)])
            sg = np.array([0, 0])

    '''
    l1 = crank length
    l2 = connecting rod length
    l3 = coupler length along connecting rod
    l4 = coupler length perpendicular to l3
    phi = orientation of fix joint from origin

    fg = Crank ground joint (on the circle with at phi angle)
    sg = Second ground joind
    First Floating Point
    '''
    fe = np.array([l1*np.cos(theta), l1*np.sin(theta)]) + fg
    #condition = (l2 + l3 > getDistance(sg,fe)) and (abs(l2 - l3) < getDistance(sg,fe))
    condition = l2 >= np.abs(fe[1])
    if not condition:
        return False, [0]

    # l2 sin(temp) = fe[1]
    temp = np.arcsin(fe[1]/l2)
    x_offset = l2*np.cos(temp)
    se1 = np.array([fe[0]+x_offset, 0])
    se2 = np.array([fe[0]-x_offset, 0])
    # Second Floating Points
    l3_inclination1 = np.arctan2(se1[1]-fe[1], se1[0]-fe[0])
    l3_inclination2 = np.arctan2(se2[1]-fe[1], se2[0]-fe[0])
    temp = getEndPoint((fe+se1)/2, l3, l3_inclination1)
    c1 = getEndPoint(temp, l4, l3_inclination1 + np.pi/2.0)
    temp = getEndPoint((fe+se2)/2, l3, l3_inclination2)
    c2 = getEndPoint(temp, l4, l3_inclination2 + np.pi/2.0)
    if all_joints:
        return True, [{'A0': fg, 'A1': sg, 'B0': fe, 'B1': se1, 'phi': theta, 'C':c1, 'theta': l3_inclination1},{'A0': fg, 'A1': sg, 'B0': fe, 'B1': se2, 'phi': theta, 'C':c2, 'theta': l3_inclination2}]
    else:
        return True, np.array([[c1[0], c1[1], l3_inclination1], [c2[0], c2[1], l3_inclination2]])

def get_grashof_2R_link_params(curve):
    ''' How to find parameters of 2R manipulators such that its workspace
        should include c curve
    '''
    n_points, dim = curve.shape
    var = np.sqrt(np.var(curve[:,0])**2 + np.var(curve[:,1])**2)
    x = np.random.normal(np.mean(curve[:,0]),2*var)
    y = np.random.normal(np.mean(curve[:,1]),2*var)
    sg = [x, y]
    r_min = np.min(np.sqrt((curve[:,0]-x)**2 + (curve[:,1]-y)**2))
    r_max = np.max(np.sqrt((curve[:,0]-x)**2 + (curve[:,1]-y)**2))
    ''' making choice here, that r1 is always grater than r2
        r1 + r2 = r_max * 1.2 (here 1.2 is given for avoiding singularity
        r1 - r2 = r_min * 0.8 (here 1.2 is given for avoiding singularity
        example:
        r1 = (r_min*0.8 + r_max*1.2)/2.0
        r2 = (r_max*1.2 - r_min*0.8)/2.0
    '''
    r_min = r_min*np.random.uniform(0.1, 0.9)
    r_max = r_max*np.random.uniform(1.1, 1.9)
    r1 = (r_min + r_max)/2.0
    r2 = (r_max - r_min)/2.0

    return sg, r1, r2


def fk_interm_step(fe, sg, l2, l3, l4, l5):
    '''
    fe is end point of crank or some motion,
    l2, l3 are the lengths of 2R manipulator reaching fe.
    l4 and l4 are coupler dimensions

    output is coupler end point and orientation
    '''
    condition = (l2 + l3 > getDistance(sg,fe)) and (abs(l2 - l3) < getDistance(sg,fe))
    if not condition:
        print('distance between sg, fe is %0.3f'%getDistance(sg,fe) + ', where l1 =%0.3f, l2=%0.3f'%(l2,l3))
    assert condition

    x_inclination = np.arctan2(sg[1]-fe[1], sg[0]-fe[0])
    x1, y1 = fe
    x2, y2 = sg
    r, R = l2, l3
    d = np.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))
    x = (d**2-r**2+R**2)/(2.0*d)
    a = 1/d * np.sqrt(4*(d**2)*(R**2)-np.power((d**2)-(r**2)+(R**2),2))
    temp = getEndPoint(fe, x, x_inclination)
    # Second Floating Points
    se1 = getEndPoint(temp, a/2.0, x_inclination + np.pi/2.0)
    se2 = getEndPoint(temp, a/2.0, x_inclination - np.pi/2.0)

    l3_inclination1 = np.arctan2(se1[1]-fe[1], se1[0]-fe[0])
    l3_inclination2 = np.arctan2(se2[1]-fe[1], se2[0]-fe[0])
    temp = getEndPoint((fe+se1)/2, l4, l3_inclination1)
    c1 = getEndPoint(temp, l5, l3_inclination1 + np.pi/2.0)
    temp = getEndPoint((fe+se2)/2, l4, l3_inclination2)
    c2 = getEndPoint(temp, l5, l3_inclination2 + np.pi/2.0)
    return c1, c2, l3_inclination1, l3_inclination2, se1, se2

def sixbar_watt_fk(theta0, r0, r1, r2, r3, r4, four_bar_params, theta, fb_type = 'fourR' , all_joints=False, grashof_sixbar=True, **kwargs):
    if kwargs is not None:
        try:
            fg = kwargs['fg']
            sg = kwargs['sg']
        except KeyError:
            if fb_type == 'sliderCrank':
                l1, l2, l3, l4, phi = four_bar_params
                fg = np.array([np.cos(phi), np.sin(phi)])
                sg = np.array([0, 0])
                success, output = slidercrank_fk(l1, l2, l3, l4, phi, theta, all_joints=all_joints, **kwargs)
            elif fb_type == 'fourR':
                l1, l2, l3, l4, l5 = four_bar_params
                fg = np.array([0, 0])
                sg = np.array([1, 0])
                success, output = fourbar_fk(l1, l2, l3, l4, l5, theta, all_joints=all_joints, **kwargs)

    if success:
        if not all_joints:
            c1 = output[0,:2]
            c2 = output[0,:2]
        else:
            c1, c2 = output['C']
    else:
        return False, [0]

    if grashof_sixbar:
        sg1, r11, r21 = get_grashof_2R_link_params(c1)
        sg2, r12, r22 = get_grashof_2R_link_params(c2)
    else:
        theta01, r01, r11, r21 = theta0, r0, r1, r2
        theta02, r02, r12, r22 = theta0, r0, r1, r2
        sg1 = np.array([r01*np.cos(theta01), r01*np.sin(theta01)])
        sg2 = np.array([r02*np.cos(theta02), r02*np.sin(theta02)])

    try:
        cp11, cp12, c_angle11, c_angle12, se11, se12 = fk_interm_step(c1, sg1, r11, r21, r3, r4)
    except AssertionError:
        print(AssertionError)
        cp11, cp12, c_angle11, c_angle12, se11, se12 = None, None, None, None, None, None
    try:
        cp21, cp22, c_angle21, c_angle22, se21, se21 = fk_interm_step(c2, sg2, r12, r22, r3, r4)
    except AssertionError:
        print(AssertionError)
        cp21, cp22, c_angle21, c_angle22, se21, se21 = None, None, None, None, None, None

    if cp11 is None and cp21 is None:
        return False, [0]

    # [{'A0': fg, 'A1': sg, 'B0': fe, 'B1': se1, 'phi': theta, 'C':c1, 'theta': l3_inclination1},{'A0': fg, 'A1': sg, 'B0': fe, 'B1': se2, 'phi': theta, 'C':c2, 'theta': l3_inclination2}]
    sixbar_output = [
            {
            'C': np.array([cp11[0], cp11[1], c_angle11]),
            'B0': c1,
            'B10': se11,
            },
            {
            'C': np.array([cp12[0], cp12[1], c_angle12]),
            'B0': c2,
            'B10': se21,
            }
            #'C3': np.array([output[1, 0], output[1, 1], output[1, 2]]),
            #'C3': np.array([cp21[0], cp21[1], c_angle21]),
            #'C4': np.array([cp22[0], cp22[1], c_angle22]),
            ]

    return True, sixbar_output
    '''
    params exlusive to sixbars
    fg = Crank ground joint (on the circle with at phi angle)
    sg = Second ground joind
    First Floating Point
    '''

def simulate_sixbar(params, timing = None, both_branches=True, only_first_curve=False, all_joints=False, fb_type='fourR', grashof_sixbar=True):
    l1,l2,l3,l4,l5, theta0, r0, r1, r2, r3, r4 = params
    fourbar_params = l1,l2,l3,l4,l5
    if fb_type == 'fourR':
        coupler_curves = simulate_fourbar(fourbar_params, all_joints=True, both_branches=False)
    elif fb_type == 'slider-crank-linkage':
        coupler_curves = simulate_fourbar(fourbar_params, fb_type='sliderCrank', all_joints=True)
    for curve, joint_data in zip(coupler_curves.curves, coupler_curves.joints):
        if grashof_sixbar:
            sg, r1, r2 = get_grashof_2R_link_params(curve)
            theta0 = np.arctan2(sg[1], sg[0])
        else:
            sg = np.array([r0*np.cos(theta0), r0*np.sin(theta0)])
        for c, joint in zip(curve, joint_data):
            try:
                cp11, cp12, c_angle11, c_angle12, se11, se12 = fk_interm_step(c[:2], sg, r1, r2, r3, r4)
            except AssertionError:
                cp11, cp12, c_angle11, c_angle12, se21, se22 = None, None, None, None, None, None

            if cp11 is None:
                return False, [0]
            else:
                joint['Cp0'] = np.array([cp11[0], cp11[1], c_angle11])
                joint['Cp1'] = np.array([cp12[0], cp12[1], c_angle12])
                joint['D0'] = sg
                joint['E0'] = se11
                joint['E1'] = se12

    params = np.array([l1,l2,l3,l4,l5, theta0, r0, r1, r2, r3, r4])

    coupler_curves.finish(only_first_curve)

    return coupler_curves, params

def simulate_sixbar_(params, timing = None, both_branches=True, only_first_curve=False, all_joints=False, fb_type='fourR', grashof_sixbar=True):
    l1,l2,l3,l4,l5, theta0, r0, r1, r2, r3, r4 = params
    fourbar_params = l1,l2,l3,l4,l5
    coupler_curves = CouplerCurves()
    circuit_changed = False
    if only_first_curve:
        both_branches = False
    if timing is None:
        timing = np.arange(0.0, 2*np.pi, np.pi/180.0)
    if timing is 'random':
        start = np.random.uniform(0, 2*np.pi)
        timing = np.arange(0.0 + start, 2*np.pi + start, np.pi/180.0)

    for theta in timing:
        success, output = sixbar_watt_fk(theta0, r0, r1, r2, r3, r4, fourbar_params, theta, all_joints=all_joints, fb_type=fb_type, grashof_sixbar=grashof_sixbar)
        if success:
            coupler_pose = np.array([[output[0]['C'][0], output[0]['C'][1], output[0]['theta']], [output[1]['C'][0], output[1]['C'][1], output[1]['theta']]])
            coupler_curves.push_point(coupler_pose)
            if all_joints:
                coupler_curves.push_joints(output)
            coupler_curves.push_point(coupler_pose)
            circuit_changed = False
        else:
            if not circuit_changed:
                coupler_curves.change_circuit()
                circuit_changed = True
    if both_branches and fb_type == 'fourR':
        circuit_changed = False
        coupler_curves.change_crank()
        for theta in timing:
            fourbar_params = l2,l1,l3,-l4,l5
            success, output = sixbar_watt_fk(theta0, r0, r1, r2, r3, r4, fourbar_params, theta, all_joints=all_joints, fb_type=fb_type, grashof_sixbar=grashof_sixbar)
            if success:
                coupler_pose = np.array([[output[0]['C'][0], output[0]['C'][1], output[0]['theta']], [output[1]['C'][0], output[1]['C'][1], output[1]['theta']]])
                coupler_curves.push_point(coupler_pose)
                if all_joints:
                    coupler_curves.push_joints(output)
                circuit_changed = False
            else:
                if not circuit_changed:
                    coupler_curves.change_circuit()
                    circuit_changed = True
    coupler_curves.finish(only_first_curve)
    return coupler_curves

def normalized_cross_corelation(s1, s2, ax=None, ax2=None, grashof=False):
    ''' Obtains similarity between two scaled signals
    s1 = sign1['path_sign']
    s2 = sign2['path_sign']
    '''

    if len(s2) >= len(s1):
        t = s1
        F = s2
    else:
        t = s2
        F = s1

    if grashof:
        F = np.concatenate((F,F))

    if ax is not None:
        ax.plot(t, 'o', ms=5, lw=2, label='Templete')
        ax.plot(F, 'v', ms=5, lw=2, label='Long Curve')

    t_mean = np.mean(t)
    v1 = []
    v2 = []
    tspan = len(t)
    t_diff_square = np.sum(np.power(t - t_mean,2))
    for u in range(len(F) - len(t) + 1):
        if(u==0):
            F_sum = np.sum(F[u:u+tspan])
            F_mean = F_sum/tspan
            num = np.sum((F[u:u+tspan]-F_mean)*(t - t_mean))
            f_diff_square = np.sum(np.power(F[u:u+tspan]-F_mean, 2))
        else:
            F_sum = F_sum + F[u+tspan-1] - F[u]
            F_mean = F_sum/tspan
            num = np.sum((F[u:u+tspan]-F_mean)*(t - t_mean))
            f_diff_square = np.sum(np.power(F[u:u+tspan]-F_mean, 2))
        v1.append(num/np.sqrt(f_diff_square*t_diff_square))
    F = F[::-1]
    for u in range(len(F) - len(t) + 1):
        if(u==0):
            F_sum = np.sum(F[u:u+tspan])
            F_mean = F_sum/tspan
            num = np.sum((F[u:u+tspan]-F_mean)*(t - t_mean))
            f_diff_square = np.sum(np.power(F[u:u+tspan]-F_mean, 2))
        else:
            F_sum = F_sum + F[u+tspan-1] - F[u]
            F_mean = F_sum/tspan
            num = np.sum((F[u:u+tspan]-F_mean)*(t - t_mean))
            f_diff_square = np.sum(np.power(F[u:u+tspan]-F_mean, 2))
        v2.append(num/np.sqrt(f_diff_square*t_diff_square))

    if ax2 is not None:
        ax2.plot(v1,'*',ms='4', label='Forward')
        ax2.plot(v2,'-',ms='4', label='Reversed')

    v1 = np.square(v1)
    v2 = np.square(v2)
    v = [np.max(v1), np.max(v2)]
    v_ = [v1, v2]
    score = np.max(v)
    order = np.argmax(v)
    offset = np.argmax(v_[order])
    output = {'score': score, 'order': order, 'offset': offset}
    if np.isnan(score):
        return {'score': -1, 'order': order, 'offset': offset}
    else:
        return output

def motion_cross_corelation(s1, s2, ax=None, ax2=None):
    """ Obtains similarity between two scaled signals of same wavelength
    s1 = sign1['motion_sign']
    s2 = sign2['motion_sign']
    """

    if len(s2) >= len(s1):
        t = s1
        F = s2
    else:
        t = s2
        F = s1
    if ax is not None:
        ax.plot(t, 'v', ms=5, lw=2, label='Templete')
        ax.plot(F, '-', ms=5, lw=2, label='Long Curve')

    t_mean = np.mean(t)
    v1 = []
    v2 = []
    tspan = len(t)
    for u in range(len(F) - len(t) + 1):
        if(u==0):
            F_sum = np.sum(F[u:u+tspan])
            F_mean = F_sum/tspan
            num = np.sum(np.power((F[u:u+tspan]-F_mean)-(t - t_mean),2))
        else:
            F_sum = F_sum + F[u+tspan-1] - F[u]
            F_mean = F_sum/tspan
            num = np.sum(np.power((F[u:u+tspan]-F_mean)-(t - t_mean),2))
        v1.append(num)

    F = F[::-1]
    for u in range(len(F) - len(t) + 1):
        if(u==0):
            F_sum = np.sum(F[u:u+tspan])
            F_mean = F_sum/tspan
            num = np.sum(np.power((F[u:u+tspan]-F_mean)-(t - t_mean),2))
        else:
            F_sum = F_sum + F[u+tspan-1] - F[u]
            F_mean = F_sum/tspan
            num = np.sum(np.power((F[u:u+tspan]-F_mean)-(t - t_mean),2))
        v2.append(num)

    if ax2 is not None:
        ax2.plot(v1,'--',ms='4', label='Forward')
        ax2.plot(v2,'-',ms='4', label='Reverse')
    v = [np.min(v1), np.min(v2)]
    v_ = [v1, v2]
    distance = np.min(v)
    order = np.argmin(v)
    offset = np.argmin(v_[order])
    output = {'distance': distance, 'order': order, 'offset': offset}
    return output

# ------------------------------------------------------------------

def planar_rigid_body_transformation_on_path(poses, t=None, phi=None, Trans=None):
    if Trans is None:
        assert(t is not None)
        assert(phi is not None)
        Trans = np.array([[np.cos(phi), -np.sin(phi), t[0]],
                      [np.sin(phi), np.cos(phi), t[1]],
                      [0, 0, 1]])

    """ poses must be in shape [batch, num, 3]
        t is translation vector = [tx, ty]
        phi is rotation angle
    """
    batch, num, dim = poses.shape
    poses_ = copy.deepcopy(poses)
    i = 0
    for traj in poses_:
        temp = copy.deepcopy(traj[:,2])
        traj[:,2] = 1
        X_new = np.matmul(Trans, traj.T)
        poses_[i] = X_new.T
        poses_[i,:,2] = temp
        i += 1
    return poses_

def planar_scaling_on_path(poses, s):
    """ poses must be in shape [batch, num, 3]
        t is translation vector = [tx, ty]
        phi is rotation angle
    """
    batch, num, dim = poses.shape
    Trans = np.array([[s, 0, 0],
                  [0, s, 0],
                  [0, 0, 1]])
    i = 0
    poses_ = copy.deepcopy(poses)
    for traj in poses_:
        temp = copy.deepcopy(traj[:,2])
        traj[:,2] = 1
        X_new = np.matmul(Trans, traj.T)
        poses_[i] = X_new.T
        poses_[i,:,2] = temp
        i += 1
    return poses_

def spatial_rigid_body_transformation_on_3d_curve(poses, R, t):
    """ poses must be in shape [batch, num, 3]
        R is rotation matrix
        t is translation vector = [tx, ty]
    """
    pass

def transform_to_planar_dual_qt(poses):
    x = poses[:,:,0]
    y = poses[:,:,1]
    theta = poses[:,:,2]
    z3 = np.sin(theta / 2);
    z4 = np.cos(theta / 2);
    z1 = 0.5 * (x * z3 - y * z4);
    z2 = 0.5 * (x * z4 + y * z3);
    return np.transpose((z1, z2, z3, z4), [1, 2, 0])

def ff_descriptor(Ks, max_harmonics=5, T=0.01):
    """ input: signature Ks should be in shape =  [batch, 101]
        output: fourier discriptors upto max_harmonics
    """
    f_descriptors = np.zeros((len(Ks), max_harmonics),dtype=np.float32)
    i = 0
    for y in Ks:
        n = len(y) # length of the signal
        k = np.arange(n)
        frq = k/T # two sides frequency range
        frq = frq[range(n/2)] # one side frequency range

        Y = np.fft.fft(y)/n # fft computing and normalization
        Y = Y[range(n/2)]
        f_descriptors[i,:] = np.abs(Y[:max_harmonics])
        i += 1
    return f_descriptors

def get_pca_transform(qx, qy, ax=None, label=''):
    """ Performs the PCA
        Return transformation matrix
    """
    cx = np.mean(qx)
    cy = np.mean(qy)
    covar_xx = np.sum((qx - cx)*(qx - cx))/len(qx)
    covar_xy = np.sum((qx - cx)*(qy - cy))/len(qx)
    covar_yx = np.sum((qy - cy)*(qx - cx))/len(qx)
    covar_yy = np.sum((qy - cy)*(qy - cy))/len(qx)
    covar = np.array([[covar_xx, covar_xy],[covar_yx, covar_yy]])
    eig_val, eig_vec= np.linalg.eig(covar)

    # Inclination of major principal axis w.r.t. x axis
    if eig_val[0] > eig_val[1]:
        phi= np.arctan2(eig_vec[1,0], eig_vec[0,0])
    else:
        phi= np.arctan2(eig_vec[1,1], eig_vec[0,1])
    # Transformation matrix T
    trans = np.array([[1, 0, -cx],
                  [0, 1, -cy],
                  [0, 0, 1]])
    rot = np.array([[np.cos(phi), np.sin(phi), 0],
                  [-np.sin(phi), np.cos(phi), 0],
                  [0, 0, 1]])
    Trans = np.matmul(rot,trans)
    if ax is not None:
        ax.plot(qx, qy, '*',ms=4, label=label)
        phi_m = np.arctan2(eig_vec[1,0], eig_vec[0,0])
        e = Ellipse(xy=[cx, cy], width=eig_val[0], height=eig_val[1], angle=phi_m*180/np.pi)
        e.set_alpha(0.5)
        ax.add_artist(e)
    max_eig = 1/np.sqrt(np.max(eig_val))
    return Trans, ax, max_eig

def get_pca_inclination(qx, qy, ax=None, label=''):
    """ Performs the PCA
        Return transformation matrix
    """
    cx = np.mean(qx)
    cy = np.mean(qy)
    covar_xx = np.sum((qx - cx)*(qx - cx))/len(qx)
    covar_xy = np.sum((qx - cx)*(qy - cy))/len(qx)
    covar_yx = np.sum((qy - cy)*(qx - cx))/len(qx)
    covar_yy = np.sum((qy - cy)*(qy - cy))/len(qx)
    covar = np.array([[covar_xx, covar_xy],[covar_yx, covar_yy]])
    eig_val, eig_vec= np.linalg.eig(covar)

    # Inclination of major principal axis w.r.t. x axis
    if eig_val[0] > eig_val[1]:
        phi= np.arctan2(eig_vec[1,0], eig_vec[0,0])
    else:
        phi= np.arctan2(eig_vec[1,1], eig_vec[0,1])

    return phi

def get_scale_factor(poses):
    '''
    poses has shape [1, num, 3]
    '''
    qx = poses[0,:,0]
    qy = poses[0,:,1]
    cx = np.mean(qx)
    cy = np.mean(qy)
    covar_xx = np.sum((qx - cx)*(qx - cx))/len(qx)
    covar_xy = np.sum((qx - cx)*(qy - cy))/len(qx)
    covar_yx = np.sum((qy - cy)*(qx - cx))/len(qx)
    covar_yy = np.sum((qy - cy)*(qy - cy))/len(qx)
    covar = np.array([[covar_xx, covar_xy],[covar_yx, covar_yy]])
    eig_val, eig_vec= np.linalg.eig(covar)

    return np.sqrt(np.sum(eig_val))

def transform_poses(x,y,theta, ax=None, label=''):
    """ Translates and rotates path part of the poses along
        Principal Directions
        input: x shape :[num, 3]
        Return: Poses => (x_transformed, y_transformed, theta_original)
    """
    #T, ax, scale_factor = get_pca_transform(x, y, ax=ax, label=label + ' : Original')
    pose = np.transpose((x, y, theta),[1,0])
    #pose = planar_scaling_on_path(poses=np.array([pose]), s=scale_factor)[0]
    #T, ax, scale_factor = get_pca_transform(pose[:,0], pose[:,1])
    #pose = planar_rigid_body_transformation_on_path(poses=np.array([pose]), Trans=T)[0]
    #pose = reform_starting_point(pose)
    #if ax is not None:
    #    ax.plot(pose[:,0], pose[:,1],'o',label=label + ' : Trans-Rotated')
    return pose

def reform_starting_point(pose):
    """ input: cupler curve trajectory of shape [num, 3]
        output: reformed, with changed starting point
    """
    leftmost = np.argmin(pose[:,0])
    if pose.shape[0] != 359:
        if (leftmost <= (pose.shape[0] - leftmost)):
            ax.plot(pose[leftmost,0], pose[leftmost,1], '*')
            ax.plot(pose[0,0], pose[0,1], '*')
            return pose
        else:
            ax.plot(pose[leftmost,0], pose[leftmost,1], '*')
            ax.plot(pose[0,0], pose[0,1], '*')
            return pose[::-1]
    else:
        new_pose = np.concatenate((pose[leftmost:],pose[:,leftmost]))
        new_pose = np.concatenate((new_pose,pose[leftmost]))

def truncate_curves(curv):
    """ curv is a list of len (num*3)
    """
    n  = len(curv)
    temp2 = []
    temp3 = []
    temp4 = []
    if n % 6 == 0:
        temp2.append(curv[:n/2])
        temp2.append(curv[n/2:])
    if n % 9 == 0:
        temp3.append(curv[:n/3])
        temp3.append(curv[n/3:2*n/3])
        temp3.append(curv[2*n/3:])
    if n % 12 == 0:
        temp4.append(curv[:n/4])
        temp4.append(curv[n/4:n/2])
        temp4.append(curv[n/2:3*n/4])
        temp4.append(curv[3*n/4:])
    return temp2, temp3, temp4

def transform_into_cylindrical_cord(poses):
    cylindrical_representation = np.zeros(poses.shape)
    cylindrical_representation[:,0] = poses[:,0]*np.cos(poses[:,2])
    cylindrical_representation[:,1] = poses[:,0]*np.sin(poses[:,2])
    cylindrical_representation[:,2] = poses[:,1]
    return cylindrical_representation

def getEndPoint(startPoint, length, angle):
    return np.array([startPoint[0] + (length * np.cos(angle)), startPoint[1] + (length * np.sin(angle))]);

def getDistance(pt1,pt2):
        return np.sqrt(np.sum(np.power(pt1-pt2,2)));

def getParams(linkage):
    link = json.loads(linkage)
    fe = np.array([link['linkageInfo'][1][3][1], link['linkageInfo'][1][4][1]])
    se = np.array([link['linkageInfo'][2][3][1], link['linkageInfo'][2][4][1]])
    l1, l2, l3 = np.sqrt(np.sum(fe**2)), np.sqrt(np.sum((se - [1,0])**2)), np.sqrt(np.sum((se - fe)**2))
    lc, ang = np.array([link['linkageInfo'][3][1][1], link['linkageInfo'][3][2][1]])
    l5 = lc*np.sin(ang)
    l4 = lc*np.cos(ang) - l3/2.0
    return [l1,l2,l3,l4,l5]
