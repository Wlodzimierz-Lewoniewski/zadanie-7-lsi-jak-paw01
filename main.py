import numpy as np
import re

def tokenizuj_tekst(tekst):
    return re.findall(r'\b\w+\b', tekst.lower())

def stworz_macierz(dokumenty, tokeny_zapytania):
    unikaty = set()
    for dokument in dokumenty:
        for slowo in dokument:
            unikaty.add(slowo)
    
    unikaty_zap = set(tokeny_zapytania)
    wszystkie_termy = unikaty.union(unikaty_zap)
    
    posortowane_termy = sorted(wszystkie_termy)
    
    macierz_incydencji = []
    for term in posortowane_termy:
        wiersz = []
        for dokument in dokumenty:
            if term in dokument:
                wiersz.append(1)
            else:
                wiersz.append(0)
        
        if term in tokeny_zapytania:
            wiersz.append(1)
        else:
            wiersz.append(0) 
        
        macierz_incydencji.append(wiersz)
    
    macierz_numpy = np.array(macierz_incydencji)
    
    return posortowane_termy, macierz_numpy

def redukcja_wymiarowosci(macierz, wymiar):
    U, s, Vt = np.linalg.svd(macierz, full_matrices=False)
    
    Sk = np.diag(s[:wymiar])
    Uk = U[:, :wymiar]
    Vk = Vt[:wymiar, :]
    
    #mnozenie
    macierz_zredukowana = Uk @ Sk @ Vk
    
    return macierz_zredukowana, Uk, Sk, Vk

def transformuj_zapytanie(wektor_zapytania, Uk, Sk):
    Sk_odwr = np.linalg.inv(Sk)
    zapytanie_zredukowane = Sk_odwr @ Uk.T @ wektor_zapytania
    return zapytanie_zredukowane

def podobienstwo_cosinusowe(zapytanie_zredukowane, dokumenty_zredukowane):
    podobienstwa = []
    for wektor_dokumentu in dokumenty_zredukowane.T:
        licznik = np.dot(zapytanie_zredukowane, wektor_dokumentu)
        mianownik = np.linalg.norm(zapytanie_zredukowane) * np.linalg.norm(wektor_dokumentu)
        podobienstwo = licznik / mianownik if mianownik != 0 else 0.0 
        podobienstwa.append(round(float(podobienstwo), 2))  
    return podobienstwa

# czesc glowna

liczba_dokumentow = int(input())
dokumenty = []
for i in range(liczba_dokumentow):
    dokument = input().strip()
    dokumenty.append(tokenizuj_tekst(dokument))
    
zapytanie = tokenizuj_tekst(input().strip())
wymiar_k = int(input())

unikalne_termy, macierz_incydencji = stworz_macierz(dokumenty, zapytanie)

macierz_dokumenty = macierz_incydencji[:, :-1]
wektor_zapytania = macierz_incydencji[:, -1]  

macierz_zredukowana, Uk, Sk, Vk = redukcja_wymiarowosci(macierz_dokumenty, wymiar_k)

dokumenty_zredukowane = Sk @ Vk

zapytanie_zredukowane = transformuj_zapytanie(wektor_zapytania, Uk, Sk)

podobienstwa = podobienstwo_cosinusowe(zapytanie_zredukowane, dokumenty_zredukowane)

print(podobienstwa)

