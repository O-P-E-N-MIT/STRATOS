import numpy as np
    
def compute_added_mass (vertices, faces, normals):
    # 3. Build X, Y, Z matrices
    
    print("Building X Y Z matrices.")

    n = faces.shape[0]
    
    X = np.zeros((3, n))
    Y = np.zeros((3, n))
    Z = np.zeros((3, n))
    
    for i in range(n):
        A = faces[i]
        X[:, i] = vertices[A, 0]
        Y[:, i] = vertices[A, 1]
        Z[:, i] = vertices[A, 2]
    
    # 4. Centroids of each face
    
    print("Computing centroids of each face.")
    
    p = np.mean(X, axis=0)
    q = np.mean(Y, axis=0)
    r = np.mean(Z, axis=0)
    
    # 5. Distance and influence matrices
    
    print("Computing distance and influence matrices.")
    
    dist = np.zeros((n, n))
    cc = np.zeros((n, n))
    ccc = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                dvec = np.array([p[i]-p[j], q[i]-q[j], r[i]-r[j]])
                dist[i, j] = np.linalg.norm(dvec)
                cc[i, j] = np.dot(dvec, normals[j])
                ccc[i, j] = cc[i, j] / (dist[i, j]**3)
    
    # 6. B, C matrices and surface area
    
    print("Computing B, C matrices and surface area.")
    
    B = np.zeros((n, n))
    C = np.zeros((n, n))
    SS = np.zeros((n, 1))
    
    for i in range(n):
        V1 = np.array([X[0, i], Y[0, i], Z[0, i]])
        V2 = np.array([X[1, i], Y[1, i], Z[1, i]])
        V3 = np.array([X[2, i], Y[2, i], Z[2, i]])
    
        # print(V1, V2, V3)
    
        S = np.cross(V2 - V1, V3 - V1) / 2
        SS[i, 0] = np.linalg.norm(S)
    
        for j in range(n):
            if j != i:
                B[j, i] = SS[i, 0] / dist[j, i]
                C[j, i] = SS[i, 0] * ccc[j, i]
    
    # 7. Delta matrix and final C matrix
    
    print("Computing delta matrix and final C matrix.")
    
    delt = np.eye(n)
    Cfinal = (2 * np.pi * delt) - C
    
    # 8. Volume and centroid
    
    print("Computing volume and centroid.")
    
    vol = np.zeros(n)
    volsx = np.zeros(n)
    volsy = np.zeros(n)
    volsz = np.zeros(n)
    
    for i in range(n):
        cvv = np.vstack((X[:, i], Y[:, i], Z[:, i]))
        vol[i] = np.linalg.det(cvv) / 6
        volsx[i] = np.sum(X[:, i]) * vol[i] / 4
        volsy[i] = np.sum(Y[:, i]) * vol[i] / 4
        volsz[i] = np.sum(Z[:, i]) * vol[i] / 4
    
    voltotal = np.sum(vol)
    xcv = np.sum(volsx) / voltotal
    ycv = np.sum(volsy) / voltotal
    zcv = np.sum(volsz) / voltotal
    
    # 9. Boundary functions
    
    print("Computing boundary functions.")
    
    xcg = p - xcv
    ycg = q - ycv
    zcg = r - zcv
    
    norm_mag = np.linalg.norm(normals, axis=1)
    
    alphaa = normals[:, 0] / norm_mag
    betaa  = normals[:, 1] / norm_mag
    gammaa = normals[:, 2] / norm_mag
    
    boundfour = gammaa * ycg - betaa * zcg
    boundfive = alphaa * zcg - gammaa * xcg
    boundsix  = betaa * xcg - alphaa * ycg
    
    # 10. Potential calculations
    
    print("Calculating potentials.")
    
    phi = [
        np.linalg.solve(Cfinal, B @ alphaa),
        np.linalg.solve(Cfinal, B @ betaa),
        np.linalg.solve(Cfinal, B @ gammaa),
        np.linalg.solve(Cfinal, B @ boundfour),
        np.linalg.solve(Cfinal, B @ boundfive),
        np.linalg.solve(Cfinal, B @ boundsix)
    ]
    
    boundary = [alphaa, betaa, gammaa, boundfour, boundfive, boundsix]
    
    # 11. Added Mass Matrix
    
    print("Computed Added Mass matrix.")
    
    M = np.zeros((6, 6))
    
    for i in range(6):
        for j in range(6):
            M[i, j] = np.sum(phi[i] * boundary[j] * SS[:, 0])
    
    AM_final = np.round(M / voltotal, 2)
    
    return AM_final