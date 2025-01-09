import cv2
import numpy as np
import math

def angle_between_lines(lineA, lineB):
    """
    Funkcja zwraca kąt w stopniach między dwiema liniami
    każda linia = ((x1,y1),(x2,y2))
    """
    # Wyznacz wektory
    (x1A, y1A), (x2A, y2A) = lineA
    (x1B, y1B), (x2B, y2B) = lineB

    # Wektory
    vA = (x2A - x1A, y2A - y1A)
    vB = (x2B - x1B, y2B - y1B)

    # Iloczyn skalarny, długości
    dot = vA[0]*vB[0] + vA[1]*vB[1]
    magA = math.sqrt(vA[0]**2 + vA[1]**2)
    magB = math.sqrt(vB[0]**2 + vB[1]**2)

    # Unikamy dzielenia przez zero
    if magA * magB == 0:
        return 180  # duży kąt, by uznać linie za "różne"

    # Kąt w radianach
    cos_ = dot / (magA * magB)
    # Zabezpieczenie przed niedokładnościami floating point
    cos_ = max(-1.0, min(1.0, cos_))

    # Przekształcenie na stopnie
    angle_deg = math.degrees(math.acos(cos_))
    return angle_deg

def midpoint(p1, p2):
    """Zwraca środek odcinka pomiędzy punktami p1, p2."""
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

def is_spot_empty(image, corners):
    """
    Naiwna funkcja, która sprawdza, czy dane miejsce parkingowe
    (zdefiniowane przez 4 rogi corners) jest puste.
    - Tutaj przykładowo liczymy średnią jasność w regionie i
      jeśli jest zbyt ciemno/jasno, możemy uznać, że stoi tam auto.
    - Realnie trzeba by użyć np. detekcji obiektów, analizy krawędzi
      czy segmentacji, by stwierdzić 'czy jest auto'.
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    # corners: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    pts = np.array([corners], dtype=np.int32)
    cv2.fillPoly(mask, pts, 255)

    # Wyciągamy tylko piksele wewnątrz czworokąta
    roi = cv2.bitwise_and(image, image, mask=mask)

    # Zamieniamy na szarość i wyliczamy średnią
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mean_val = cv2.mean(roi_gray, mask=mask)[0]  # Średnia jasność

    # Przykładowe proste kryterium:
    # jeśli średnia jasność jest poniżej/ powyżej pewnego progu, 
    # zakładamy, że tam "coś" stoi (samochód), w przeciwnym wypadku pusto.
    # Te wartości (80, 200) są przykładowymi progami.
    if 80 < mean_val < 200:
        # "prawdopodobnie" pusto
        return True
    else:
        return False

def group_lines_to_parking_spots(lines):
    """
    Funkcja-baza do grupowania linii w czworokąty/równoległoboki.
    Bardzo uproszczona heurystyka:
      1. Dzielimy linie na 'prawie pionowe' i 'prawie poziome' w zależności od kąta.
      2. Potem parujemy linie pionowe-pionowe i poziome-poziome,
         sprawdzając, czy przecinają się w "rozsądnych" miejscach.
    Zwraca listę czworokątów (każdy czworokąt: lista 4 punktów w kolejności).
    """
    vertical = []
    horizontal = []

    # Rozdzielamy linie na pionowe i poziome (według progu kąta z osią x)
    # Dla większej dokładności można sprawdzać kąt linii względem osi x
    # i dać np. threshold = 20-30 stopni
    for l in lines:
        (x1, y1, x2, y2) = l[0]
        angle_rad = math.atan2((y2 - y1), (x2 - x1))
        angle_deg = abs(math.degrees(angle_rad))
        if angle_deg < 30 or angle_deg > 150:
            # Blisko poziomu (0°) lub ~180°
            horizontal.append(((x1,y1),(x2,y2)))
        else:
            # Blisko pionu (90°)
            vertical.append(((x1,y1),(x2,y2)))

    # Tu zaczyna się trudniejsza część –
    # łączenie linii w równoległoboki. Przykład bardzo naiwny:
    # 1. Bierzemy parę linii poziomych, sprawdzamy, czy są do siebie
    #    w miarę równoległe i dość blisko, by tworzyć 'górę' i 'dół'.
    # 2. Bierzemy parę linii pionowych, analogicznie.
    # 3. Sprawdzamy przecięcia tych linii, by dostać 4 rogi.

    spots = []

    # (Naiwne) - bierzemy co drugą parę z horizontal i vertical
    # aby zademonstrować tworzenie "czworokątów"  
    # W realnym zastosowaniu należałoby wykonać bardziej
    # zaawansowany algorytm (np. klastrowanie według pozycji, itp.)

    h_pairs = []
    for i in range(len(horizontal)):
        for j in range(i+1, len(horizontal)):
            # Sprawdzamy kąt, by się upewnić, że są prawie równoległe
            ang = angle_between_lines(horizontal[i], horizontal[j])
            if ang < 10:  # można wybrać inny próg
                h_pairs.append((horizontal[i], horizontal[j]))

    v_pairs = []
    for i in range(len(vertical)):
        for j in range(i+1, len(vertical)):
            ang = angle_between_lines(vertical[i], vertical[j])
            if ang < 10:
                v_pairs.append((vertical[i], vertical[j]))

    # Teraz tworzymy prostokąty (A, B, C, D) – każde to przecięcia par linii
    for (h1, h2) in h_pairs:
        for (v1, v2) in v_pairs:
            # h1 i h2 to dwie linie poziome
            # v1 i v2 to dwie linie pionowe
            # Wyznaczamy 4 przeciecia:
            #    * A = intersection(h1, v1)
            #    * B = intersection(h1, v2)
            #    * C = intersection(h2, v2)
            #    * D = intersection(h2, v1)
            # w idealnym świecie to da nam narożniki równoległoboku (czasem trapezu).

            A = line_intersection(h1, v1)
            B = line_intersection(h1, v2)
            C = line_intersection(h2, v2)
            D = line_intersection(h2, v1)

            # Jeżeli wszystkie zwracają sensowne punkty (nie None),
            # tworzymy czworokąt.
            if A and B and C and D:
                # Można sprawdzić np. czy te punkty leżą
                # w granicach obrazu itd.
                polygon = [A, B, C, D]
                spots.append(polygon)

    return spots

def line_intersection(line1, line2):
    """
    Zwraca punkt przecięcia dwóch odcinków,
    w postaci (x, y) lub None, jeśli się nie przecinają
    w obrębie tych konkretnych odcinków.
    line1 = ((x1,y1),(x2,y2))
    line2 = ((x3,y3),(x4,y4))
    """
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2

    # Oparte na równaniach parametrycznych linii
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None

    # Punkt przecięcia w układzie rzeczywistym:
    intersect_x = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    intersect_y = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom

    # Teraz trzeba sprawdzić, czy punkt leży "na odcinkach", a nie na całych liniach
    # Warunek: intersect_x musi być pomiędzy min i max x1,x2 oraz x3,x4 (analogicznie y)
    if (min(x1, x2) <= intersect_x <= max(x1, x2) and
            min(y1, y2) <= intersect_y <= max(y1, y2) and
            min(x3, x4) <= intersect_x <= max(x3, x4) and
            min(y3, y4) <= intersect_y <= max(y3, y4)):
        return (int(intersect_x), int(intersect_y))
    else:
        return None


def main():
    # 1) Wczytanie obrazu
    image = cv2.imread("empty.jpg")
    if image is None:
        print("Nie udało się wczytać obrazu!")
        return

    orig = image.copy()

    # 2) Wstępna obróbka obrazu (skala szarości, rozmycie, Canny)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)

    # 3) Detekcja linii (HoughLinesP)
    lines = cv2.HoughLinesP(edges,
                            rho=1,
                            theta=np.pi/180,
                            threshold=70,
                            minLineLength=40,
                            maxLineGap=10)

    # 4) Grupowanie linii w kształty (równoległoboki)
    if lines is not None:
        # lines to tablica Nx1x4 (x1,y1,x2,y2)
        spots = group_lines_to_parking_spots(lines)
    else:
        spots = []

    # 5) Sprawdzanie, czy miejsce jest puste i rysowanie
    #    Dla uproszczenia zakładamy, że jeżeli jest puste, rysujemy zielony obrys
    for spot in spots:
        # spot to lista 4 punktów: [(xA,yA), (xB,yB), (xC,yC), (xD,yD)]
        if is_spot_empty(orig, spot):
            pts = np.array(spot, dtype=np.int32)
            cv2.polylines(orig, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        else:
            pts = np.array(spot, dtype=np.int32)
            cv2.polylines(orig, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

    # 6) Wyświetlenie wyniku
    cv2.imshow("Parking spots detection", orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
