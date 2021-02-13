from imageio import imread, imwrite
import numpy as np
import matplotlib.pyplot as plt


def lag_gauss_filter(sigma):
    """
    Lager et Gauss-filter etter gitte spesifikasjoner.
    Args:
        n: Dimensjonen til filteret (n x n)
        sigma: Standardavviket som brukes til utregning
    Returns:
        gauss_filter: Normalisert Gauss-filter
    """
    n = np.round(1+8*sigma)
    gauss_filter = np.zeros((n,n))
    # Senter av filteret tilsvarer 0, der det går fra -a til a vertikalt og horisontalt
    a = int((n-1)/2)
    sum = 0
    for i in range(-a,a+1):
        for j in range(-a,a+1):
            h = np.exp(-(i**2+j**2)/(2*sigma**2))
            # Legger til a i indeksering for å få riktig posisjon i filteret
            gauss_filter[i+a,j+a] = h
            sum += h

    return gauss_filter/sum

def konvolusjon(image, filter):
    """
    Konvolverer inn-bildet med gitt filter. Verdier utenfor bildet blir paddet
    nærmeste pikselverdi i inn-bildet. Ut-bildet har samme størrelse som
    inn-bildet.
    Args:
        image: Inn-bildet som skal konvolveres
        filter: Filteret som skal konvolvere inn-bildet
    Returns:
        output: Konvolvert versjon av inn-bildet
    """
    M,N = image.shape
    m,n = filter.shape

    filter = np.rot90(np.rot90(filter)) # Filteret roteres 180 grader for konvolusjon
    a = int((m-1)/2)    # antall rader vi må nullutvide med oppe og nede
    b = int((n-1)/2)    # antall kolonner vi må nullutvide med til høyre og venstre

    padded_img = np.pad(image,((a,a),(b,b)),mode='edge') # Utvider bildet med nærmeste pikselverdi
    output = np.zeros((M,N))

    for i in range(M):
        for j in range(N):
            # Ganger filteret med overlappende verdier i bildet og summerer
            verdi = padded_img[i:(i+2*a+1),j:(j+2*b+1)]*filter
            output[i,j] = np.sum(verdi)
    return output


def gradient_magnitude_vinkel(bilde):
    """
    Regner ut gradientmagnitude og gradientvinkel til bildet.
    Gradientmagnituden blir reskalert til intervallet [0,255]
    Args:
        bilde: bildet som skal analyseres
    Returns:
        M,theta: Gradientmagnitude og gradientvinkel
    """
    # Symmetrisk 1D-operator
    hx = np.array([[0,1,0],[0,0,0],[0,-1,0]])
    hy = np.array([[0,0,0],[1,0,-1],[0,0,0]])

    gx = konvolusjon(bilde,hx)
    gy = konvolusjon(bilde,hy)

    M = np.sqrt(gx**2+gy**2)
    theta = np.arctan2(gy,gx)
    # Transformering fra radianer til grader
    theta = theta*(180/np.pi)

    return reskaler_til_8bit(M),theta


def kant_tynning(M_1,theta):
    """
    Tynner kantene til inn-bildet ved å sjekke om
    Args:
        M_1  : Gradientmagnitude-bildet som skal tynnes
        theta: Gradientvinkler til pikslene i bildet
    Returns:
        M: Det ferdig tynnede bildet
    """
    M = np.copy(M_1)
    m,n = theta.shape
    M_pad = np.pad(M,1,'constant') # Null-padder M for å sammenligne 8-naboer

    for i in range(m):
        for j in range(n):
            s,t = i+1,j+1 # gjelpevariabler til M_pad for å iterere over bildet
            retning = theta[i,j]
            # Horisontal kant
            if (np.abs(retning) <= 22.5) | (np.abs(retning) > 157.5):
                if (M_pad[s+1,t] > M_pad[s,t]) | (M_pad[s-1,t] > M_pad[s,t]):
                    M[i,j] = 0
            # -45-graders kant
            elif (22.5 < retning <= 67.5) | (-112.5 > retning >= -157.5):
                if (M_pad[s+1,t+1] > M_pad[s,t]) | (M_pad[s-1,t-1] > M_pad[s,t]):
                    M[i,j] = 0
            # Vertikal kant
            elif (67.5 < np.abs(retning) <= 112.5):
                if (M_pad[s,t+1] > M_pad[s,t]) | (M_pad[s,t-1] > M_pad[s,t]):
                    M[i,j] = 0
            # +45-graders kant
            elif (112.5 < retning <= 157.5) | (-22.5 > retning >= -67.5):
                if (M_pad[s-1,t+1] > M_pad[s,t]) | (M_pad[s+1,t-1] > M_pad[s,t]):
                    M[i,j] = 0
    return M

def reskaler_til_8bit(bilde):
    max = np.max(bilde)
    min = np.min(bilde)
    # Lineær normalisering av verdiene til å ligge mellom 0 og 255 (8-bit)
    bilde = ((255-0)/(max-min))*(bilde-min)
    return bilde


def hystereseterskling(g_N, Tl, Th):
    """
    Det tynnede bildet blir tersklet ved å merke intensiteter over Th og
    så hente 8-naboer til de merkede som ligger mellom Tl og Th til konvergens
    Args:
        g_N: bilde med tynnede kanter
        Tl: lav-terskel for hysterese
        Th: høy-terskel for hysterese
    Returns:
        g_NH: Hysteresetersklet bilde
    """
    M,N = g_N.shape
    # Finner intensiteter over Th og setter resten 0
    g_NH = np.where(g_N >= Th, g_N, 0)
    # Plukker ut intensiteter mellom Tl og Th
    g_NL = np.where((Tl <= g_N) & (g_N < Th), g_N, 0)

    # Indeksene til merkede kanter lagres i "merket"
    merket = np.nonzero(g_NH)

    # Loopen kjører så lenge vi har nye merkede kanter
    while(len(merket[1]) > 0):
        nye_merkede = [[],[]]
        for i,j in zip(merket[0],merket[1]):
            # s- og t-loopen sjekker 8-nabolaget i g_NL[i,j]
            for s in range(-1,2):
                for t in range(-1,2):
                    x,y = i+s,j+t
                    if (0 <= x < M) & (0 <= y < N):
                        if g_NL[x,y] > 0:
                            g_NH[x,y] = g_NL[x,y]
                            g_NL[x,y] = 0
                            # Lagrer indeks til den nye merkede pikselen
                            nye_merkede[0].append(x)
                            nye_merkede[1].append(y)
        # De nye pikslene som ble merket som kant, sjekkes for naboer
        # i neste iterasjon
        merket = nye_merkede
    return g_NH


def canny():
    sigma = 5
    Tl = 40
    Th = 80

    img_celle = imread("cellekjerner.png", as_gray = True)

    filter = lag_gauss_filter(sigma)

    print("Filtrering...")
    filtrert_img = konvolusjon(img_celle,filter)

    print("Regner ut M og theta...")
    M,theta = gradient_magnitude_vinkel(filtrert_img)

    print("Kant-tynning...")
    tynnet_M = kant_tynning(M,theta)

    print("Hysterese...")
    tersklet = hystereseterskling(tynnet_M,Tl,Th)
    # Resultatet lagres som 8-bit bilde
    tersklet = (np.round(tersklet)).astype(np.uint8)

    print("FERDIG")

    # Bildet skrives til fil
    imwrite("resultat.png",tersklet)

if __name__ == '__main__':
    canny()
