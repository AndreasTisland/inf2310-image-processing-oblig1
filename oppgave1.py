import numpy as np
import matplotlib.pyplot as plt
from imageio import imread, imwrite


def middelverdi(p):
    return sum(i*p[i] for i in range(len(p)))


def standardavvik(p):
    G = len(p)
    return sum((i**2)*p[i] for i in range(G)) - sum(i*p[i] for i in range(G))**2


def normalisert_histogram(image):
        M,N = image.shape
        hist = histogram(image)
        return hist/(M*N)


def histogram(image):
    hist = np.zeros(256)
    M,N = image.shape

    for i in range(M):
        for j in range(N):
            hist[int(image[i][j])] += 1

    return hist


def finn_koeffisienter():
    """
    Finner koeffisientene som skal brukes til den affine transformasjonen
    ved å mappe punkter fra øyne og munn til en maske.

    Returns:
        Matrise med a- og b-koeffisientene
    """
    X = np.array([[88,68,109],[84,120,129],[1,1,1]])    # Punkter fra portrett
    Y = np.array([[260,260,443],[170,341,257],[1,1,1]]) # Punkter fra maske

    A = np.dot(Y,np.linalg.inv(X))

    return A


def affin_transform(inn_bilde, maske, transform):
    """
    Transformerer input-bilde til å matche maskens posisjoner til øyne og munn.

    Args:
        image:      input-bilde som skal transformeres
        mask:       masken som input-bildet skal matches med
        transform:  matrise med transform-koeffisienter
    Returns:
        Nytt transformert ut-bilde
    """
    M,N = inn_bilde.shape
    m,n = maske.shape
    output = np.zeros(maske.shape)
    for i in range(M):
        for j in range(N):
            pos = np.array([[i],[j],[1]]) # Posisjon hos input som skal transformeres
            nypos = np.dot(transform,pos) # Ny transformert posisjon i ut-bildet
            x,y = np.round(nypos[0]),np.round(nypos[1])
            if (0 <= x < m) & (0 <= y < n):
                # Hvis ny posisjon er innenfor utbildet, kopieres itensiteten i
                # inn-bildet til den nye posisjonen i ut-bildet
                output[int(x),int(y)] = inn_bilde[i][j]

    return output

def kontrast_standardisering(image, mu=127, o=64):
    """
    Standardiserer kontrast ved å transformere bildet til å få ny middelverdi
    og standardavvik på 127 og 64. Deretter normaliseres bildet for å kunne
    lagres som 8-bit.
    Args:
        image:  bildet som skal standardiseres
        mu:     ny middelverdi
        o:      nytt standardavvik
    Returns:
        Bildet med standardisert kontrast
    """
    p = normalisert_histogram(image)
    mu_T = middelverdi(p)
    o_T = standardavvik(p)

    a = o_T/o
    b = mu_T-a*mu

    standardisert_bilde = a*image+b

    max = np.max(standardisert_bilde)
    min = np.min(standardisert_bilde)
    # Lineær normalisering av verdiene til å ligge mellom 0 og 255 (8-bit)
    standardisert_bilde = ((255-0)/(max-min))*(standardisert_bilde-min)

    return standardisert_bilde


def baklengs_transform_narmeste_nabo(ut_bilde, inn_bilde, transform):
    """
    Baklengs transform med nærmeste-nabo-interpolasjon. Hver piksel i ut-bildet
    får intensiteten til den nærmeste pikselen den mappes til i inn-bildet.
    Args:
        ut_bilde: det transformerte bildet som skal interpoleres
        inn_bilde: original-bildet som det skal hentes intensiteter fra
        transform: transformen fra inn-bildet til ut-bildet, denne inverteres
    Returns:
        Nærmeste-nabo-interpolert versjon av ut-bildet
    """
    A_inv = np.linalg.inv(transform)
    M,N = ut_bilde.shape
    narmeste_nabo = np.zeros(ut_bilde.shape)
    for i in range(M):
        for j in range(N):
            pos = np.array([[i],[j],[1]])
            nypos = np.dot(A_inv,pos)
            x,y = np.round(nypos[0]),np.round(nypos[1])
            narmeste_nabo[i,j] = inn_bilde[int(x),int(y)]

    return narmeste_nabo

def baklengs_transform_bilinear(ut_bilde, inn_bilde, transform):
    """
    Baklengs transform med bilineær interpolasjon. For hver piksel i ut-bildet
    brukes en lineær formel for å finne beste intensitet basert på de 4 nærmeste
    piksler den mappes til i inn-bildet.
    Args:
        ut_bilde: det transformerte bildet som skal interpoleres
        inn_bilde: original-bildet som det skal hentes intensiteter fra
        transform: transformen fra inn-bildet til ut-bildet, denne inverteres
    Returns:
        Bilineær interpolasjon av ut-bildet
    """
    A_inv = np.linalg.inv(transform)
    M,N = ut_bilde.shape
    bilinear = np.zeros(ut_bilde.shape)
    for i in range(M):
        for j in range(N):
            pos = np.array([[i],[j],[1]])
            nypos = np.dot(A_inv,pos)
            x,y = nypos[0],nypos[1]
            x0,x1 = int(np.floor(x)),int(np.ceil(x))
            y0,y1 = int(np.floor(y)),int(np.ceil(y))
            dx = x-x0
            dy = y-y0
            p = inn_bilde[x0,y0] + (inn_bilde[x1,y0]-inn_bilde[x0,y0])*dx
            q = inn_bilde[x0,y1] + (inn_bilde[x1,y1]-inn_bilde[x0,y1])*dx

            bilinear[i,j] = p + (q-p)*dy

    return bilinear


def test():
    img_portrett = imread("portrett.png", as_gray = True)
    img_mask = imread("geometrimaske.png", as_gray = True)

    A = finn_koeffisienter()
    kontrast_standardisert_bilde = kontrast_standardisering(img_portrett)

    standardisert_bilde = affin_transform(kontrast_standardisert_bilde,img_mask,A)

    narmeste_nabo = baklengs_transform_narmeste_nabo(standardisert_bilde,kontrast_standardisert_bilde,A)

    bilinear = baklengs_transform_bilinear(standardisert_bilde, kontrast_standardisert_bilde,A)

    # Lagrer bilder som 8-bit
    kontrast_standardisert_bilde = (np.round(kontrast_standardisert_bilde)).astype(np.uint8)

    standardisert_bilde = (np.round(standardisert_bilde)).astype(np.uint8)

    narmeste_nabo = (np.round(narmeste_nabo)).astype(np.uint8)

    bilinear = (np.round(bilinear)).astype(np.uint8)

    # Bilder skrives til fil
    imwrite("kontrast_standardisert.png",kontrast_standardisert_bilde)
    imwrite("standardisert_bilde.png",standardisert_bilde)
    imwrite("narmeste_nabo.png",narmeste_nabo)
    imwrite("bilinear.png",bilinear)


if __name__ == '__main__':
    test()
