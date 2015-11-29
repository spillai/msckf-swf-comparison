import numpy as np

def crossMat(vec): 
    """
    Computes the cross-product matrix of a 3x1 vector
    
    """ 
    return np.float64([[0, -vec[2], vec[1]], 
                       [vec[2], 0, -vec[0]], 
                       [-vec[1], vec[0], 0]])

def axisAngleToRotMat(psi): 
    """    
    Converts an axis-angle rotation vector into a rotation matrix
    
    psi               - axis-angle rotation vector
    """

    nm = np.linalg.norm(psi)
    cp = np.cos(nm)
    sp = np.sin(nm)
    pnp = psi / np.linalg.norm(psi)

    Psi = cp.dot(np.eye(3)) + (1 - cp).dot(pnp.dot(pnp.T)) - sp.dot(crossMat(pnp))

    return Psi

def crossMatToVec(crossMat): 
    """
    Extracts a 3x1 vector from a cross-product matrix
    
    crossMat           - 3x3 skew-symmetric matrix
    """
    return np.float64([crossMat[2,1], crossMat[0,2], crossMat[1,0]])

def buildUpdateQuat(deltaTheta): 
    """
    Builds the update quaternion from the minimally parametrized update
    See Indirect Kalman Filter for 3D Attitude Estimation (Roumeliotis)
    """
    
    deltaq = 0.5 * deltaTheta
    
    checkNorm = np.dot(deltaq.T, deltaq)
    
    if checkNorm > 1: 
        updateQuat = np.hstack([deltaq, 1])
        updateQuat = updateQuat / np.sqrt(1 + checkNorm)
    else: 
        updateQuat = np.hstack([deltaq, np.sqrt(1 - checkNorm)])
        
    updateQuat = updateQuat / np.linalg.norm(updateQuat)
    return updateQuat

def omegaMat(omega): 
    """
    Computes the Omega matrix of a 3x1 vector, omega
    FIX (returns 4x4 matrix)
    """
    return np.float64([[-crossMat(omega), omega], 
                       [-omega, 0]])

def quatInv(quat): 
    """    
    Computes the inverse (or conjugate) of a unit quaternion
    using the {i,j,k,1} convention
    """
    if np.abs(np.linalg.norm(quat) - 1) > 1e-6 :
        raise RuntimeError('Input quaternion must be unit-length')

    return np.float64([-quat[:3], quat[3]])

def quatLeftComp(quat): 
    """    
    Computes the left-hand compound operator form of a quaternion
    using the {i,j,k,1} convention (q^+ in Tim's book)
    """

    return np.vstack([
        np.hstack([quat[3] * np.eye(3) - crossMat(quat[:3]), quat[:3].reshape(-1,1)]), 
        np.float64([-quat[:3], quat[3]])])

def quatRightComp(quat): 
    """
    Computes the right-hand compound operator form of a quaternion
    using the {i,j,k,1} convention (q^\oplus in Tim's book)
    """

    return np.vstack([
        np.hstack([quat[3] * np.eye(3) + crossMat(quat[:3]), quat[:3].reshape(-1,1)]), 
        np.float64([-quat[:3], quat[3]])])

def quatMult(quat1, quat2): 
    """
    Multiplies two quaternions using the {i,j,k,1} convention
    """

    return quatLeftComp(quat1).dot(quat2)

def quatToRotMat(quat): 
    """
    Converts a quaternion into a 3x3 rotation matrix
    using the {i,j,k,1} convention
    """
    
    if ( np.abs(np.linalg.norm(quat) - 1) > 1e-6 ):
        if np.abs(np.linalg.norm(quat) - 1) > 0.1:
            raise RuntimeWarning('''Input quaternion is not unit-length'''
                                 '''. norm(q) = %f. Re-normalizing.'''
                                 % np.linalg.norm(quat))
        
        quat = quat / np.linalg.norm(quat)
    
    R = (quatRightComp(quat).T).dot(quatLeftComp(quat))
    C = renormalizeRotMat( R[:3,:3] )
    return C

def renormalizeRotMat(C): 
    """
    Enforce det(C) = 1 by finding the nearest orthogonal matrix
    """
    [U,S,V] = np.linalg.svd(C)
    C_unitary = U.dot( np.eye(S.shape[0] ).dot( V.T ) )
    return C_unitary
                       
