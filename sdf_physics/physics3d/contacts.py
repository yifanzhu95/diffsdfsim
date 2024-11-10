#
# Copyright 2024 Max-Planck-Gesellschaft
# Code author: Michael Strecke, michael.strecke@tuebingen.mpg.de
# Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import torch
from diffworld.diffsdfsim.lcp_physics.physics.contacts import ContactHandler, OdeContactHandler
from diffworld.utils import triangle_point_distance_and_normal_batched
from pytorch3d.transforms import quaternion_apply, quaternion_invert, quaternion_multiply
from scipy.spatial.qhull import ConvexHull, QhullError
from pykeops.torch import LazyTensor
from torch.nn.functional import normalize
import time
from .bodies import SDF3D, Body3D
from .utils import Defaults3D
from icecream import ic
import numpy as np
import trimesh

def KMeans(x, K=10, Niter=10, verbose=False):
    """Implements Lloyd's algorithm for the Euclidean metric."""
    start = time.time()
    N, D = x.shape  # Number of samples, dimension of the ambient space

    c = x[:K, :].clone()  # Simplistic initialization for the centroids

    x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
    c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):
        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        c /= Ncl  # in-place division to compute the average

    if verbose:  # Fancy display -----------------------------------------------
        if use_cuda:
            torch.cuda.synchronize()
        end = time.time()
        print(
            f"K-means for the Euclidean metric with {N:,} points in dimension {D:,}, K = {K:,}:"
        )
        print(
            "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                Niter, end - start, Niter, (end - start) / Niter
            )
        )

    return cl, c

def _overlap(b1, b2):
    v1 = b1.get_surface()[0]
    v2 = b2.get_surface()[0]

    v1_b2 = quaternion_apply(quaternion_invert(b2.rot), v1 - b2.pos)
    v2_b1 = quaternion_apply(quaternion_invert(b1.rot), v2 - b1.pos)

    ov_v1_b2 = torch.any(torch.all((-b2.scale <= v1_b2) & (v1_b2 <= b2.scale), dim=1))
    ov_v2_b1 = torch.any(torch.all((-b1.scale <= v2_b1) & (v2_b1 <= b1.scale), dim=1))
    return ov_v1_b2 and ov_v2_b1


def _frank_wolfe(b1, b2, eps=Defaults3D.EPSILON, tol=Defaults3D.TOL):
    verts, faces = b1.get_surface()
    # Convert vertices to b2's coordinate frame for easier SDF querying
    verts = quaternion_apply(quaternion_invert(b2.rot), verts - b2.pos)

    x = verts[faces].mean(dim=1).squeeze()

    centr_sdfs, centr_grads = b2.query_sdfs(x)
    rads = x.new_zeros(x.shape[0])
    for i in range(3):
        new_rads = (x - verts[faces[:, i]]).norm(dim=1)
        rads[new_rads > rads] = new_rads[new_rads > rads]

    cand_mask = (centr_sdfs < rads + eps) & (centr_grads.norm(dim=1) > 1e-12)
    pqr = verts[faces[cand_mask]]
    if not torch.any(cand_mask):
        return pqr.new_empty((0, 3)), torch.nonzero(cand_mask, as_tuple=False).squeeze(1)

    sdfs = b2.query_sdfs(pqr.reshape(-1, 3), return_grads=False).reshape(pqr.shape[0], -1)
    abc = sdfs.new_zeros(sdfs.shape)
    inds = sdfs.argmin(dim=1)
    x = pqr[torch.arange(pqr.shape[0]), inds, :]
    abc[torch.arange(abc.shape[0]), inds] = 1.

    for iter in range(32):
        sdfs, grads = b2.query_sdfs(x)

        dpqr = pqr @ grads.unsqueeze(2)

        ind = dpqr.argmin(dim=1)
        s = pqr[torch.arange(pqr.shape[0]), ind.squeeze()]

        gamma = 2.0 / (iter + 2.0)

        impr = ((x - s).unsqueeze(1) @ grads.unsqueeze(2)).squeeze(2)
        gamma = gamma * (impr.abs() > tol)
        if torch.all(gamma == 0) or torch.any(sdfs < -tol):
            # Stop if improvement for all points too small or
            # if we found a point that will cause step rejection
            break

        x = (1.0 - gamma) * x + gamma * s
        abc *= (1.0 - gamma)
        abc[torch.arange(abc.shape[0]), ind.squeeze()] += gamma.squeeze()

    if isinstance(b1, SDF3D):
        # Push x to actual surface from triangle
        x_b1 = (b1.verts[b1.faces[cand_mask]] * abc.unsqueeze(2)).sum(dim=1)
        sdfs1, grads1 = b1.query_sdfs(x_b1)
        x = x - sdfs1.unsqueeze(1) * quaternion_apply(quaternion_multiply(quaternion_invert(b2.rot), b1.rot), grads1)
    sdfs = b2.query_sdfs(x, return_grads=False)

    contact_mask = (sdfs <= eps)  # & (sdfs - sdfs.min() < 1e-5)
    cand_mask[cand_mask.clone()] &= contact_mask

    return abc[contact_mask], torch.nonzero(cand_mask, as_tuple=False).squeeze(1)


def _filter_contacts(normals, p1, eps=Defaults3D.EPSILON):
    contact_inds = torch.arange(normals.shape[0], device=normals.device)
    if normals.shape[0] <= 1:
        return contact_inds

    # At SDF singularities, normals might be zero
    valid_mask = normals.norm(dim=1) > 1e-12
    normals = normals[valid_mask]
    p1 = p1[valid_mask]
    contact_inds = contact_inds[valid_mask]

    clusters = []
    while normals.shape[0] > 0:
        n = normals[0]

        angles = torch.acos(torch.min(normals @ n, n.new_tensor(1.)))
        cluster_mask = (angles < 1e-2)

        c_p1s = p1[cluster_mask]
        c_inds = contact_inds[cluster_mask]

        clusters.append((c_p1s, c_inds))

        normals = normals[~cluster_mask]
        p1 = p1[~cluster_mask]
        contact_inds = contact_inds[~cluster_mask]

    clusters_filtered = []
    # Filter further points based by generating a (possibly lower-dimensional) convex hull
    for p1, cluster_inds in clusters:
        completed = False
        ps = p1.detach()
        while not completed:
            if ps.shape[1] > 1:
                try:
                    hull = ConvexHull(ps.cpu())
                    inds = torch.tensor(hull.vertices).long().to(normals.device)
                    completed = True
                except QhullError:
                    # Qhull didn't work, likely because the points only span a lower-dim space
                    # => remove dim with smallest variance
                    var = ps.var(dim=0)
                    mask = torch.ones(ps.shape[1]).bool()
                    mask[var.argmin()] = False
                    ps = ps[:, mask]
            else:
                # If we reduced the points to 1D convex hull is just min and max
                ps_min, ps_argmin = ps.min(0)
                ps_max, ps_argmax = ps.max(0)
                if ps_max - ps_min > eps:
                    inds = torch.stack([ps.argmin(), ps.argmax()])
                else:
                    inds = torch.stack([ps.argmin()])
                completed = True

        clusters_filtered.append(cluster_inds[inds])

    filtered_inds = torch.empty((0), dtype=torch.int64, device=normals.device)
    for cluster_inds in clusters_filtered:
        filtered_inds = torch.cat([filtered_inds, cluster_inds], dim=0)

    return filtered_inds


def _compute_contacts(b1, b2, abc, contact_inds, eps=Defaults3D.EPSILON, detach_contact_b2=True):
    if contact_inds.nelement() > 0:
        # verts, faces = b1.get_surface()
        verts, faces = b1.verts, b1.faces

        cp_b1 = (verts[faces[contact_inds]] * abc.unsqueeze(2)).sum(dim=1)
        if isinstance(b1, SDF3D):
            # Contact points on triangle might not be on surface, correct for this.
            dists1, normals1 = b1.query_sdfs(cp_b1)
            cp_b1 = cp_b1 - dists1.unsqueeze(1) * normals1
            dists1, normals1 = b1.query_sdfs(cp_b1)

        contact_points = quaternion_apply(b1.rot, cp_b1) + b1.pos

        if detach_contact_b2:
            # TODO: This detach() stop gradients needed e.g. for block tower example (when contact points need to be
            #       moved consistently for optimization).
            cp_b2 = quaternion_apply(quaternion_invert(b2.rot), contact_points - b2.pos).detach()
        else:
            cp_b2 = quaternion_apply(quaternion_invert(b2.rot), contact_points - b2.pos)

        dists2, normals2 = b2.query_sdfs(cp_b2)

        if isinstance(b1, SDF3D):
            laplacian1 = cp_b1.new_zeros(cp_b1.shape[0])
            for i in range(3):
                shift = cp_b1.new_zeros(3)
                shift[i] = eps
                laplacian1 += b1.query_sdfs(cp_b1 + shift, return_grads=False) - 2 * dists1 \
                              + b1.query_sdfs(cp_b1 - shift, return_grads=False)

            laplacian2 = cp_b2.new_zeros(cp_b2.shape[0])
            for i in range(3):
                shift = cp_b2.new_zeros(3)
                shift[i] = eps
                laplacian2 += b2.query_sdfs(cp_b2 + shift, return_grads=False) - 2 * dists2 \
                              + b2.query_sdfs(cp_b2 - shift, return_grads=False)

            stable_mask = laplacian2.abs() < laplacian1.abs()

            normals = quaternion_apply(b2.rot, normals2) * stable_mask.unsqueeze(1) \
                      - quaternion_apply(b1.rot, normals1) * (~stable_mask).unsqueeze(1)
        else:
            normals = quaternion_apply(b2.rot, normals2)
        dists = dists2

        # p2 = contact_points - b2.pos - dists.unsqueeze(1) / 2. * normals
        p2 = quaternion_apply(b2.rot, cp_b2 - dists2.unsqueeze(1) * normals2)
        p1 = quaternion_apply(b1.rot, cp_b1)
        pen = -dists

        return normals, p1, p2, pen

    return abc.new_tensor([]), abc.new_tensor([]), abc.new_tensor([]), abc.new_tensor([])


class FWContactHandler(ContactHandler):
    def __init__(self):
        self.debug_callback = OdeContactHandler()

    def __call__(self, args, geom1, geom2):
        # self.debug_callback(args, geom1, geom2)

        if geom1 in geom2.no_contact:
            return
        world = args[0]

        b1 = world.bodies[geom1.body]
        b2 = world.bodies[geom2.body]

        assert (isinstance(b1, SDF3D) and isinstance(b2, Body3D)) or (isinstance(b2, SDF3D) and isinstance(b1, Body3D))

        if isinstance(b1, SDF3D) and isinstance(b2, SDF3D):
            if not _overlap(b1, b2):
                return

            # TODO: compare to using only one direction
            validstep = self._search_contacts(geom1, geom2, world)
            if validstep:
                self._search_contacts(geom2, geom1, world)
        elif isinstance(b1, SDF3D):
            self._search_contacts(geom2, geom1, world)
        else:
            self._search_contacts(geom1, geom2, world)

        # world.contacts_debug = world.contacts  # XXX

    @staticmethod
    def _search_contacts(geom1, geom2, world):
        b1 = world.bodies[geom1.body]
        b2 = world.bodies[geom2.body]

        assert (isinstance(b1, Body3D))
        assert (isinstance(b2, SDF3D))
        with torch.no_grad():
            abc, contact_inds = _frank_wolfe(b1, b2, world.eps, world.tol)
            normals, p1, p2, pens = _compute_contacts(b1, b2, abc, contact_inds,
                                                      detach_contact_b2=world.detach_contact_b2)
            if torch.all(pens <= world.tol):
                filtered_inds = _filter_contacts(normals, p1, eps=world.eps)

        if torch.all(pens <= world.tol):
            normals, p1, p2, pens = _compute_contacts(b1, b2, abc[filtered_inds], contact_inds[filtered_inds],
                                                      detach_contact_b2=world.detach_contact_b2)

        pts = []
        for normal, pt1, pt2, pen in zip(normals, p1, p2, pens):
            pts.append((normal, pt1, pt2, pen))
        for p in pts:
            world.contacts.append((p, geom1.body, geom2.body))

        return torch.all(pens <= world.tol)


class SDFGSDiffContactHandler(ContactHandler):
    """Differentiable contact handler, operations to calculate contact manifold
    are done in autograd.
    """
    def __init__(self):
        self.debug_callback = OdeContactHandler()

    def __call__(self, args, geom1, geom2):
        # self.debug_callback(args, geom1, geom2)
        # world is added a list of contacts, where each contact is tupe
        # (normals, p1, p2, pen)
        # normal is contact normal relative to b2 (outward of b2)
        # Calulated based off of the deeper penetration point
        # p1 and p2 are contact points on b1 and b2 relative to b1 and b2, in global frame
        # pen (>0) is the pen depth of b1's points in b2, negates dist

        # if geom1 in geom2.no_contact:
        #     return
        
        world = args[0]

        b1 = world.bodies[geom1.body]
        b2 = world.bodies[geom2.body]

        types = [b1.type, b2.type]
        if 'obj' not in types:
            return 

        if 'robot' in types and 'obj' in types:
            ## TODO implement robot primitive geometry collision handling
            return

        if b1.type == 'obj' and b2.type == 'obj':
            sdf_body = b1
            sdf_index = 1
            gs_body = b2
            gs_index = 2

        if 'obj' in types and 'terrain' in types:
            # no sdf for terrain, only use GS
            if b1.type == 'obj':
                sdf_body = b1
                sdf_index = 1
                gs_body = b2
                gs_index = 2
            else:
                sdf_body = b2
                sdf_index = 2
                gs_body = b1
                gs_index = 1

        # GS xyz point position in sdf frame, scaled
        xyz_sap_frame = quaternion_apply(quaternion_invert(sdf_body.rot), gs_body.get_gs() - sdf_body.pos)*sdf_body.scale_tensor
        padding = world.configs['sdf_collision_detection_padding']

        # import open3d as o3d
        # mesh_box = o3d.geometry.TriangleMesh.create_box(width=0.1,
        #                                         height=0.2,
        #                                         depth=0.1).translate((-0.05,-0.1, 0.01))
        # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        #     size=0.6, origin=[0,0,0])
        # p= o3d.geometry.PointCloud()
        # tmp = gs_body.get_gs().cpu().detach().numpy()
        # p.points = o3d.utility.Vector3dVector(tmp)
        # o3d.visualization.draw_geometries([p, mesh_box, mesh_frame])
        
        BB_mask = torch.logical_and(xyz_sap_frame[:,0] < (1 + padding), xyz_sap_frame[:,0] > (-1 - padding))
        BB_mask = torch.logical_and(BB_mask,xyz_sap_frame[:,1] < (1 + padding))
        BB_mask = torch.logical_and(BB_mask,xyz_sap_frame[:,1] > (-1 - padding))
        BB_mask = torch.logical_and(BB_mask,xyz_sap_frame[:,2] < (1 + padding))
        BB_mask = torch.logical_and(BB_mask,xyz_sap_frame[:,2] > (-1 - padding))
        
 
        xyz_sap_frame = xyz_sap_frame[BB_mask]
        sdfs, normals = sdf_body.sdf.forward_torch(xyz_sap_frame)
        normals = normalize(normals, dim=1)
        sdfs = sdfs/sdf_body.scale_tensor
        contact_mask = (sdfs <= world.eps)[:,0]
        sdfs = sdfs[contact_mask]
        normals = normals[contact_mask]
        xyz_sap_frame = xyz_sap_frame[contact_mask]
        if len(sdfs) <= world.configs['N_contact_cluster']:  
            pens = -sdfs
            gs_pts = (gs_body.get_gs()[BB_mask])[contact_mask] - gs_body.pos
            sdf_pts = (gs_body.get_gs()[BB_mask])[contact_mask]  - sdf_body.pos
        else:
            grid_origin = torch.min(xyz_sap_frame, axis = 0)[0]
            grid_size = world.configs['grid_size']*sdf_body.scale_tensor

            side_sizes = torch.max(xyz_sap_frame, axis = 0)[0] - grid_origin
            voxels = torch.ceil(side_sizes/grid_size)
            indeces = []
            xyz_copy = xyz_sap_frame.clone() - grid_origin
            xyz_copy /= side_sizes
            xyz_copy *= voxels
            xyz_copy = torch.ceil(xyz_copy)

            xyz_copy = xyz_copy.detach().cpu().numpy().astype(int)
            voxels = voxels.detach().cpu().numpy().astype(int) + 1
            for i in range(voxels[0]):
                for j in range(voxels[1]):
                    for k in range(voxels[2]):
                        idx =np.where((xyz_copy == [i,j,k]).all(axis=1))[0]
                        if idx.size > 0:
                            indeces.append(idx[0])
            sdfs = sdfs[indeces]
            pens = -sdfs
            normals = normals[indeces]
            gs_pts = ((gs_body.get_gs()[BB_mask])[contact_mask])[indeces] - gs_body.pos
            sdf_pts = ((gs_body.get_gs()[BB_mask])[contact_mask])[indeces]- sdf_body.pos

            # Use K means clustering
            # xyz_sdf_frame_copy = xyz_sdf_frame[contact_mask].detach().clone()
            # xyz_sdf_frame =  xyz_sdf_frame[contact_mask]
            # cl, c = KMeans(xyz_sdf_frame_copy, world.configs['N_contact_cluster'])
            # clusterred_pt_indeces = []
            # ic(c)
            # exit()
            # for d in range(world.configs['N_contact_cluster']):
            #     centroid = c[d]
            #     #ic(centroid.shape)
            #     pts = xyz_sdf_frame_copy[cl == d]
            #     dists = torch.cdist(centroid.unsqueeze(0), pts)
                
            #     _, idx = torch.min(dists, dim = 1)
            #     #ic(idx)
            #     clusterred_pt_indeces.append(idx)
            # sdfs = sdfs[clusterred_pt_indeces]
            # pens = -sdfs
            
            #pens.retain_grad()
            #world.pens = pens

            # normals = normals[clusterred_pt_indeces]
            # gs_pts = ((gs_body.get_gs()[BB_mask])[contact_mask])[clusterred_pt_indeces] - gs_body.pos
            # sdf_pts = ((gs_body.get_gs()[BB_mask])[contact_mask])[clusterred_pt_indeces]- sdf_body.pos
    
        pts = []
        if sdf_index == 2:
            for normal, pt1, pt2, pen in zip(normals, gs_pts, sdf_pts, pens):
                pts.append((normal, pt1, pt2, pen))
        elif sdf_index == 1:
            normals *= -1 
            for normal, pt1, pt2, pen in zip(normals, sdf_pts, gs_pts, pens):
                pts.append((normal, pt1, pt2, pen))

        for p in pts:
            world.contacts.append((p, geom1.body, geom2.body))

class SaPDiffContactHandler(ContactHandler):
    """Differentiable contact handler, operations to calculate contact manifold
    are done in autograd.
    """
    def __init__(self):
        self.debug_callback = OdeContactHandler()

    def __call__(self, args, geom1, geom2):
        ## Is this needed
        # if geom1 in geom2.no_contact:
        #     return
        
        world = args[0]

        b1 = world.bodies[geom1.body]
        b2 = world.bodies[geom2.body]

        types = [b1.type, b2.type]
        if 'obj' not in types:
            return 

        if 'robot' in types and 'obj' in types:
            ## TODO implement robot primitive geometry collision handling
            return

        if b1.type == 'obj' and b2.type == 'obj':
            sap_body = b1
            sap_index = 1
            other_body = b2
            other_index = 2

        if 'obj' in types and 'terrain' in types:
            if b1.type == 'obj':
                sap_body = b1
                sap_index = 1
                other_body = b2
                other_index = 2
            else:
                sap_body = b2
                sap_index = 2
                other_body = b1
                other_index = 1

        # GS xyz point position in sdf frame, scaled
        xyz_sap_frame = quaternion_apply(quaternion_invert(sap_body.rot), other_body.get_pointsnormals()[0] - sap_body.pos)*sap_body.sap_scale \
                        - sap_body.sap_offset
        padding = world.configs['collision_detection_padding']

        # import open3d as o3d
        # mesh_box = o3d.geometry.TriangleMesh.create_box(width=0.1,
        #                                         height=0.2,
        #                                         depth=0.1).translate((-0.05,-0.1, 0.01))
        # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        #     size=0.6, origin=[0,0,0])
        # p= o3d.geometry.PointCloud()
        # tmp = gs_body.get_gs().cpu().detach().numpy()
        # p.points = o3d.utility.Vector3dVector(tmp)
        # o3d.visualization.draw_geometries([p, mesh_box, mesh_frame])
        
        # sap is [0, 1)
        BB_mask = torch.logical_and(xyz_sap_frame[:,0] < (1 + padding), xyz_sap_frame[:,0] > (0 - padding))
        BB_mask = torch.logical_and(BB_mask,xyz_sap_frame[:,1] < (1 + padding))
        BB_mask = torch.logical_and(BB_mask,xyz_sap_frame[:,1] > (0 - padding))
        BB_mask = torch.logical_and(BB_mask,xyz_sap_frame[:,2] < (1 + padding))
        BB_mask = torch.logical_and(BB_mask,xyz_sap_frame[:,2] > (0 - padding))
        
        xyz_sap_frame = xyz_sap_frame[BB_mask]
        psr_grid = sap_body.psr_grid.squeeze().squeeze()
        psr_normal =  sap_body.psr_normal.squeeze().squeeze()

        xyz_sap_grid = torch.floor(xyz_sap_frame.clone()*world.configs['SaP_grid_res']).to(torch.int).cpu()
        # only use one point in each grid
        unique_rows, unique_mask, counts = torch.unique(
            xyz_sap_grid, 
            dim=0, 
            return_inverse=True, 
            return_counts=True
        )
        xyz_sap_frame = xyz_sap_frame[unique_mask]
        xyz_sap_grid = xyz_sap_grid[unique_mask]

        sdfs = psr_grid[xyz_sap_grid[:,0], xyz_sap_grid[:,1], xyz_sap_grid[:,2]].unsqueeze(-1) #unitless sdfs
        normals = psr_normal[xyz_sap_grid[:,0], xyz_sap_grid[:,1], xyz_sap_grid[:,2], :]
        sdfs = sdfs/sap_body.sap_scale
        contact_mask = (sdfs <= world.eps)[:,0].cpu()

        #contact_mask = (sdfs <= 0)[:,0]
        sdfs = sdfs[contact_mask]
        normals = normals[contact_mask]
        xyz_sap_frame = xyz_sap_frame[contact_mask]
        pens = -sdfs
        other_pts = (((other_body.get_pointsnormals()[0][BB_mask]))[unique_mask])[contact_mask]- other_body.pos
        sap_pts = (((other_body.get_pointsnormals()[0][BB_mask]))[unique_mask])[contact_mask] - sap_body.pos
        #print(pens)
        pts = []
        if sap_index == 2:
            for normal, pt1, pt2, pen in zip(normals, other_pts, sap_pts, pens):
                pts.append((normal, pt1, pt2, pen))
        elif sap_index == 1:
            normals *= -1 
            for normal, pt1, pt2, pen in zip(normals, sap_pts, other_pts, pens):
                pts.append((normal, pt1, pt2, pen))

        for p in pts:
            world.contacts.append((p, geom1.body, geom2.body))


        return
    

class SaPMeshDiffContactHandler(ContactHandler):
    """Differentiable contact handler, operations to calculate contact manifold
    are done in autograd.
    """
    def __init__(self):
        self.debug_callback = OdeContactHandler()

    def __call__(self, args, geom1, geom2):
        """
        confused!! Is normal out of b2 or b1???
        """
        world = args[0]

        b1 = world.bodies[geom1.body]
        b2 = world.bodies[geom2.body]
        types = [b1.type, b2.type]

        if 'obj' not in types:
            return 
        use_other_body_normal = False
        if 'robot' in types and 'obj' in types:
            use_other_body_normal = True
            if b1.type == 'obj':
                sap_body = b1
                sap_index = 1
                other_body = b2
            else:
                sap_body = b2
                sap_index = 2
                other_body = b1

        if b1.type == 'obj' and b2.type == 'obj':
            sap_body = b1
            sap_index = 1
            other_body = b2

        
        if 'obj' in types and 'terrain' in types:
            use_other_body_normal = True
            if b1.type == 'obj':
                sap_body = b1
                sap_index = 1
                other_body = b2
            else:
                sap_body = b2
                sap_index = 2
                other_body = b1

        # world frame
        xyz, other_normals = other_body.get_pointsnormals()
        
        #ic(torch.max(xyz, axis = 0), torch.min(xyz, axis = 0))

        padding = world.configs['collision_detection_padding']
        obj_scale = sap_body.scale_tensor
        # sap is [0, 1)
        sap_pos = sap_body.pos
        BB_mask = torch.logical_and(xyz[:,0] < 1/obj_scale*(1 + padding) + sap_pos[0], xyz[:,0] > 1/obj_scale*(-1 - padding)+ sap_pos[0])
        BB_mask = torch.logical_and(BB_mask,xyz[:,1] < 1/obj_scale*(1 + padding)+ sap_pos[1])
        BB_mask = torch.logical_and(BB_mask,xyz[:,1] > 1/obj_scale*(-1 - padding)+ sap_pos[1])
        BB_mask = torch.logical_and(BB_mask,xyz[:,2] < 1/obj_scale*(1 + padding)+ sap_pos[2])
        BB_mask = torch.logical_and(BB_mask,xyz[:,2] > 1/obj_scale*(-1 - padding)+ sap_pos[2])
        xyz = xyz[BB_mask]


        if use_other_body_normal:
            other_normals = other_normals[BB_mask]
        if len(xyz) < 1:
            return
        
        (v,f,n, _) = sap_body.get_mesh()
        f_numpy = f.squeeze(0).detach().cpu().numpy().astype(int)
        v_numpy = v.squeeze(0).detach().cpu().numpy()
        xyz_numpy = xyz.detach().cpu().numpy()
        trimesh_mesh = trimesh.Trimesh(vertices=v_numpy, faces=f_numpy)
        signed_distances = trimesh.proximity.signed_distance(trimesh_mesh, xyz_numpy)


        closest_points, distances, face_indices = trimesh.proximity.closest_point(trimesh_mesh, xyz_numpy)
        pt_indeces = f_numpy[face_indices, :]
        verteces = v[0,pt_indeces,:]
        normals = n[0,pt_indeces,:]
        _, normals, torch_dist, isnan = triangle_point_distance_and_normal_batched(
                xyz.double(),\
                verteces.double(), \
                normals.double(), \
                world.configs['norm_padding'])
        signs = torch.from_numpy(np.sign(signed_distances)*-1).to(xyz.device)
        sdfs = torch_dist*signs
        sdfs = sdfs.double()
        contact_mask = (sdfs <= world.eps).cpu()
        contact_mask = contact_mask & ~isnan.cpu()
        #contact_mask = (sdfs <= 0)[:,0]
        sdfs = sdfs[contact_mask]
        xyz= xyz[contact_mask]
        if use_other_body_normal:
            normals = other_normals[contact_mask]
        else:
            normals = normals[contact_mask]
        normals = normals.float()
        if len(xyz) < 1:
            return

        if len(sdfs) <= world.configs['N_contact_cluster']:  
            pens = -sdfs.unsqueeze(-1)
            other_pts = (other_body.get_pointsnormals()[0][BB_mask])[contact_mask]- other_body.pos
            sap_pts = (other_body.get_pointsnormals()[0][BB_mask])[contact_mask] - sap_body.pos

            tmp = (other_body.get_pointsnormals()[0][BB_mask])[contact_mask]
        else:
            def grid_cluster_3d(points, cell_size):
                # Compute the minimum values for each dimension
                min_values, _ = torch.min(points, dim=0)
                # Shift all points to be non-negative
                shifted_points = points - min_values
                # Compute grid indices for each point
                grid_indices = (shifted_points / cell_size).long()
                # Use numpy.unique on the grid indices directly
                unique_rows, indices = np.unique(grid_indices.cpu().detach().numpy(), axis=0, return_index=True)
                return indices
            contact_cluster_grid_size = world.configs.get('contact_cluster_grid_size', 0.005)
            mask = grid_cluster_3d(xyz, 0.005) #0.005
            sdfs = sdfs[mask]
            pens = -sdfs.unsqueeze(-1)
            normals = normals[mask]
            other_pts = ((other_body.get_pointsnormals()[0][BB_mask])[contact_mask])[mask] - other_body.pos
            sap_pts= ((other_body.get_pointsnormals()[0][BB_mask])[contact_mask])[mask]- sap_body.pos
            tmp = ((other_body.get_pointsnormals()[0][BB_mask])[contact_mask])[mask]
            # ic(torch.min(xyz[mask], dim = 0))
            # print(f'N of contact poitns before and after: {before_N} and {after_N}')
        # ic(tmp)
        # import open3d as o3d
        # o3d_pcd1 = o3d.geometry.PointCloud()
        # o3d_pcd1.points = o3d.utility.Vector3dVector(v.squeeze(0).detach().cpu().numpy())
        # o3d_pcd1.paint_uniform_color([1,0,0])

        # o3d_pcd2 = o3d.geometry.PointCloud()
        # o3d_pcd2.points = o3d.utility.Vector3dVector(tmp.detach().cpu().numpy())
        # o3d_pcd2.paint_uniform_color([0,1,0])

        # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[-0.3360,  0.3794, -0.0172])
        # o3d.visualization.draw_geometries([coord_frame,o3d_pcd2,o3d_pcd1]) #
        #ic(b1.type, b2.type, b1.pos, b2.pos)
        #exit()
        #ic(torch.min(other_pts, axis = 0),b1.type, b2.type, other_body.pos)
        #TODO can do contact clustering
        #ic(torch.max(pens).cpu().detach().numpy(), len(pens),b1.type, b2.type)
            
        # min_index = torch.argmin(normals[:,2])
        max_index = torch.argmax(normals[:,2])
        #if 'robot' in types and 'obj' in types:
            # ic(normals[min_index],b1.type, b2.type)
            #ic(normals[max_index], xyz[max_index])
            #ic(torch.max(pens).cpu().detach().numpy())
            #ic(normals)
            #ic(use_other_body_normal)
            #ic(b1.pos - b2.pos)
        # if 'terrain' in types and 'obj' in types:
        #     # ic(normals)
        #     ic(torch.max(pens).cpu().detach().numpy())
        #     max, indeces = torch.max(pens, dim = 0)
        #     ic(other_pts[indeces[0]])
        #     ic(normals[indeces[0]])
        #     # ic(torch.min(other_pts, axis = 0))
        #     # ic(torch.max(pens).cpu().detach().numpy(), len(pens))
        # #     ic(normals)
        pts = []
        normals = normals.double()
        if sap_index == 2:
            if use_other_body_normal:
                normals *= -1
            for normal, pt1, pt2, pen in zip(normals, other_pts, sap_pts, pens):
                pts.append((normal, pt1, pt2, pen))
        elif sap_index == 1:
            if not use_other_body_normal:
                normals *= -1 
            for normal, pt1, pt2, pen in zip(normals, sap_pts, other_pts, pens):
                pts.append((normal, pt1, pt2, pen))
            # ic(normal, pt1, pt2, pen)
        #ic(torch.max(normals, axis = 0), b1.type, b2.type)
        for p in pts:
            world.contacts.append((p, geom1.body, geom2.body))
        return