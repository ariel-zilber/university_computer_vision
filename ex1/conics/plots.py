import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
from matplotlib.patches import Ellipse as  Ellipse
from matplotlib.patches import Circle as Circle
from numpy.linalg import lstsq



#  since an exact 0 was not found in the circle case
#  I assumed  an approximation for a coefficient B to be considers 0
from numpy.polynomial import Polynomial

EPSILON_B_COEFFICIENT=5.0e-8
EPLISION_DISCRMENANT=1e-12


def read_image(file_path):
    return  cv2.imread(file_path)



def get_all_edge_points(img):
    """
    Given all the edged points on an image using edge detection

    :param edged_img:
    :return:
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 100, 300)

    points=[]

    for x in range(0,edged.shape[0]):
        for y in range(0,edged.shape[1]):
                if(edged[x][y]!=0):
                    points.append([x,y])
    return points


def get_conics_coefficients(points, f=1.0):
    """
    Solve for the coefficients of a conic given five points in Numpy array

    :return the coefficients satisfing equation
        a*x^2 + b*x*y + c*y^2 + d*x + e*y +f=0
    """

    x = points[:, 0]
    y = points[:, 1]
    if max(x.shape) < 5:
        raise ValueError('Need >= 5 points to solve for conic section')

    A = np.vstack([x**2, x * y, y**2, x, y]).T
    fullSolution = lstsq(A, f * np.ones(x.size),rcond=None)
    (a, b, c, d, e) = fullSolution[0]
    return (a, b, c, d, e, f)







def print_coefficents(a,b,c,d,e,f):
    print("A=",a)
    print("B=",b)
    print("C=",c)
    print("D=",d)
    print("E=",e)
    print("F=",f)


def axes():
    plt.axhline(0, alpha=.1)
    plt.axvline(0, alpha=.1)
def ellipse_example():

    # print the type
    print("ELLIPSE:")

    img=read_image('ellipse.png')
    points=get_all_edge_points(img)

    # get 5 points at 5 different event places on image
    cords=[points[int(cord)] for cord in np.linspace(0,len(points)-1,5)]
    sample_points=np.array(cords)
    a,b,c,d,e,f=get_conics_coefficients(sample_points)

    # display the coeffienct
    print_coefficents(a,b,c,d,e,f)

    discriminant=b**2-4*a*c
    print("discriminant:",discriminant)

    # represents an ellipse
    assert discriminant<0

    # assert circle B is in range
    assert EPSILON_B_COEFFICIENT < np.abs(b)




    # get fitted ellipse
    ellipse_info=cv2.fitEllipse(np.array(points))

    axes = ellipse_info[1]
    centers=ellipse_info[0]
    ellipse=Ellipse(xy=(centers),width=axes[0],height=axes[1],edgecolor='r', fc='None', lw=2)

    # plot the ellipse
    plt.figure()
    ax = plt.gca()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.add_patch(ellipse)
    ax.set_aspect('equal')
    ax.autoscale()
    plt.show()



def circle_example():

    # print the type
    print("CIRCLE:")

    img=read_image('circle.png')
    points=get_all_edge_points(img)

    # get 5 points at 5 different event places on image
    cords=[points[int(cord)] for cord in np.linspace(0,len(points)-1,5)]
    sample_points=np.array(cords)
    a,b,c,d,e,f=get_conics_coefficients(sample_points)

    # display the coeffienct
    print_coefficents(a,b,c,d,e,f)

    discriminant=b**2-4*a*c
    print("discriminant:",discriminant)

    # represents an ellipse
    assert discriminant<0

    # assert circle B is in range of epsillon
    assert EPSILON_B_COEFFICIENT > np.abs(b)

    # get fitted circle
    ellipse_info=cv2.fitEllipse(np.array(points))
    axes = ellipse_info[1]
    centers=ellipse_info[0]
    ellipse=Ellipse(xy=(centers),width=axes[0],height=axes[1],edgecolor='r', fc='None', lw=2)

    # plot the circle
    plt.figure()
    ax = plt.gca()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.add_patch(ellipse)
    ax.set_aspect('equal')
    ax.autoscale()
    plt.show()


def hyperbola_example():
    # print the type
    print("HYPERBOLA:")
    img=read_image('hyperbola.png')

    edged=get_all_edge_points(img)



    selected_points=np.array([
        [116,170],
        [141,267],
        [137,442],
        [310,399],
        [355,123]
            ])

    a,b,c,d,e,f=get_conics_coefficients(selected_points)

    # display the coeffienct
    print_coefficents(a,b,c,d,e,f)

    discriminant=b**2-4*a*c
    print("discriminant:",discriminant)

    # represents an hyperbola
    assert discriminant>0 and abs(discriminant)>EPLISION_DISCRMENANT

    # fit the data to the hypberbola
    x=np.array([p[0] for p in edged if p[1]<=200])
    y=[p[1] for p in edged if p[1]<=200]
    p = Polynomial.fit(x, y, 2)
    y_new=p(x)
    plt.scatter(y_new,x,color='black')

    x=np.array([p[0] for p in edged if p[1]>200])
    y=[p[1] for p in edged if p[1]>200]
    p = Polynomial.fit(x, y, 2)
    y_new=p(x)
    plt.scatter(y_new,x,color='black')


    plt.show()

def parabola_example():
    # print the type
    print("PARABOLA:")
    img=read_image('parabola.png')

    edged=get_all_edge_points(img)

    # manually entered:
    points=np.array([[165, 143], [183, 92], [221, 22], [183, 442], [161, 155]])

    a,b,c,d,e,f=get_conics_coefficients(points)

    # display the coeffienct
    print_coefficents(a,b,c,d,e,f)

    discriminant=b**2-4*a*c

    print("discriminant:",discriminant)

    # should be 0
    assert   abs(discriminant)<EPLISION_DISCRMENANT

    x=np.array([p[0] for p in edged ])
    y=[p[1] for p in edged ]
    p = Polynomial.fit(x, y, 2)
    y_new=p(x)
    plt.scatter(y_new,x,color='black')
    plt.show()




if __name__ == '__main__':
   ellipse_example()
   circle_example()
   hyperbola_example()
   parabola_example()




