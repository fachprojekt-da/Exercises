import math 
import numpy as np
from corpus import CorpusLoader
from features import BagOfWords, WordListNormalizer
from visualization import hbar_plot
import itertools

def aufgabe2():

    # Laden des Brown Corpus 
    CorpusLoader.load()
    brown = CorpusLoader.brown_corpus()
    brown_categories = brown.categories()

    # In dieser Aufgabe sollen unbekannte Dokumente zu bekannten Kategorien
    # automatisch zugeordnet werden. 
    # 
    # Die dabei erforderlichen numerischen Berechnungen lassen sich im Vergleich
    # zu einer direkten Implementierung in Python erheblich einfacher mit 
    # NumPy / SciPy durchfuehren. Die folgende Aufgabe soll Ihnen die Unterschiede
    # anhand eines kleinen Beispiels verdeutlichen.   
    #
    # Geben Sie fuer jede Katgorie des Brown Corpus die durchschnittliche Anzahl
    # von Woertern pro Dokument aus. Bestimmen Sie auch die Standardabweichung.
    # Stellen Sie diese Statistik mit einem bar plot dar. Verwenden Sie dabei 
    # auch Fehlerbalken (siehe visualization.hbar_plot)

    # Berechnen Sie Mittelwert und Standardabweichung jeweils:
    #
    #  - nur mit Python Funktion
    #    hilfreiche Funktionen: sum, float, math.sqrt, math.pow
    #
    #  - mit NumPy
    #    hilfreiche Funktionen: np.array, np.mean, np.std
    #
    # http://docs.python.org/3/library/math.html
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.std.html

    for val in brown_categories:
        print(f'Category {val}: ')
        mean = len(brown.words(categories=val)) / len(brown.fileids(val))
        print('Mittelwert: ' + str(mean))
        documents = [doc for doc in brown.fileids(val)]
        variance = sum([math.pow((len(brown.words(fileids=doc))-mean),2) for doc in documents])/len(documents)
        standard_deviation = math.sqrt(variance)
        print('Standardabweicheung: ' + str(standard_deviation))
        print('\n')

    for val in brown_categories:
        print(f'Category {val}: ')
        documents_length = [len(brown.words(fileids=doc)) for doc in brown.fileids(val)]
        print('Länge der einzelnen Dokumente: ' + str(documents_length))
        mean = np.mean(np.array(documents_length))
        standard_deviation = np.std(np.array(documents_length))
        print('Mittelwert mit Numpy: ' + str(mean))
        print('Standardabweichung mit Numpy: ' + str(standard_deviation))


    raise NotImplementedError('Implement me')
    
    # ********************************** ACHTUNG **************************************
    # Die nun zu implementierenden Funktionen spielen eine zentrale Rolle im weiteren 
    # Verlauf des Fachprojekts. Achten Sie auf eine effiziente und 'saubere' Umsetzung. 
    # Verwenden Sie geeignete Datenstrukturen und passende Python Funktionen.
    # Wenn Ihnen Ihr Ansatz sehr aufwaendig vorkommt, haben Sie vermutlich nicht die
    # passenden Datenstrukturen / Algorithmen / (highlevel) Python / NumPy Funktionen
    # verwendet. Fragen Sie in diesem Fall!
    #
    # Schauen Sie sich jetzt schon gruendlich die Klassen und deren Interfaces in den
    # mitgelieferten Modulen an. Wenn Sie Ihre Datenstrukturen von Anfang an dazu 
    # passend waehlen, erleichtert dies deren spaetere Benutzung. Zusaetzlich bieten 
    # diese Klassen bereits etwas Inspiration fuer Python-typisches Design, wie zum 
    # Beispiel Duck-Typing.
    #
    # Zu einigen der vorgebenen Intefaces finden Sie Unit Tests in dem Paket 'test'. 
    # Diese sind sehr hilfreich um zu ueberpruefen, ob ihre Implementierung zusammen
    # mit anderen mitgelieferten Implementierungen / Interfaces funktionieren wird.
    # Stellen Sie immer sicher, dass die Unit tests fuer die von Ihnen verwendeten 
    # Funktionen erfolgreich sind. 
    # Hinweis: Im Verlauf des Fachprojekts werden die Unit Tests nach und nach erfolg-
    # reich sein. Falls es sie zu Beginn stoert, wenn einzelne Unit Tests fehlschlagen
    # koennen Sie diese durch einen 'decorator' vor der Methodendefinition voruebergehend
    # abschalten: @unittest.skip('')
    # https://docs.python.org/3/library/unittest.html#skipping-tests-and-expected-failures
    # Denken Sie aber daran sie spaeter wieder zu aktivieren.
    #
    # Wenn etwas unklar ist, fragen Sie!     
    # *********************************************************************************
    
    
    #
    # Klassifikation von Dokumenten
    #
    # Nachdem Sie sich nun mit der Struktur und den Eigenschaften des Brown
    # Corpus vertraut gemacht haben, soll er die Datengrundlage fuer die 
    # Evaluierung von Algorithmen zur automatischen Klassifikation von 
    # Dokumenten bilden.
    # In der Regel bestehen diese Algorithmen aus drei Schritten:
    #  - Vorverarbeitung
    #  - Merkmalsberechnung
    #  - Klassifikation
    #
    # Bei der Anwendung auf Dokumente (Texte) werden diese Schritte wie folgt 
    # umgesetzt:
    #  - Vorverarbeitung: Filterung von stopwords und Abbildung von Woertern auf
    #                     Wortstaemme.
    #  - Merkmalsberechnung: Jedes Dokument wird numerisch durch einen Vektor 
    #                        repraesentiert (--> NumPy), der moeglichst die
    #                        bzgl. der Klassifikation bedeutungsunterscheidenden 
    #                        Informationen enthaehlt.
    #  - Klassifikation: Jedem Merkmalsvektor (Dokument) wird ein Klassenindex 
    #                    (Kategorie) zugeordnet.
    #                        
    # Details finden Sie zum Beispiel in:
    # http://www5.informatik.uni-erlangen.de/fileadmin/Persons/NiemannHeinrich/klassifikation-von-mustern/m00-www.pdf (section 1.3)
    #
    # Eine sehr verbreitete Merkmalsrepraesentation fuer (textuelle) Dokumente sind
    # sogenannte Bag-of-Words.
    # Dabei wird jedes Dokument durch ein Histogram (Verteilung) ueber Wortfrequenzen
    # repraesentiert. Man betrachtet dazu das Vorkommen von 'typischen' Woertern, die
    # durch ein Vokabular gegeben sind.
    #
    # Bestimmen Sie ein Vokabular, also die typischen Woerter, fuer den Brown Corpus.
    # Berechnen Sie dazu die 500 haeufigsten Woerter (nach stemming und Filterung von
    # stopwords und Satzzeichen)
    
    raise NotImplementedError('Implement me')
    
    
    # Berechnen Sie Bag-of-Words Repraesentationen fuer jedes Dokument des Brown Corpus.
    # Verwenden Sie absolute Frequenzen.
    # Speichern Sie die Bag-of-Word Repraesentationen fuer jede Kategorie in einem 
    # 2-D NumPy Array. Speichern Sie den Bag-of-Words Vektor fuer jedes Dokument in
    # einer Zeile, so dass das Array (ndarray) folgende Dimension hat:
    #   |Dokument_kat| X |Vokabular|
    # 
    # |Dokument_kat| entspricht der Anzahl Dokumente einer Kategorie.
    # |Vokabular| entspricht der Anzahl Woerter im Vokabular (hier 500).
    #
    # Eine einfache Zuordnung von Kategorie und Bag-of-Words Matrix ist durch ein 
    # Dictionary moeglich.
    #
    # Implementieren Sie die Funktion BagOfWords.category_bow_dict im Modul features.

    raise NotImplementedError('Implement me')
    
    # Um einen Klassifikator statistisch zu evaluieren, benoetigt man eine Trainingsstichprobe
    # und eine Teststichprobe der Daten die klassifiziert werden sollen. 
    # Die Trainingsstichprobe benoetigt man zum Erstellen oder Trainieren des Klassifikators.
    # Dabei werden in der Regel die Modellparameter des Klassifikators statistisch aus den 
    # Daten der Traingingsstichprobe geschaetzt. Die Klassenzugehoerigkeiten sind fuer die 
    # Beispiele aus der Trainingsstichprobe durch so genannte Klassenlabels gegeben.
    # 
    # Nachdem man den Klassifikator trainiert hat, interessiert man sich normalerweise dafuer
    # wie gut er sich unter realen Bedingung verhaelt. Das heisst, dass der Klassifikator bei
    # der Klassifikation zuvor unbekannter Daten moeglichst wenige Fehler machen soll.
    # Dies simuliert man mit der Teststichprobe. Da auch fuer jedes Beispiel aus der Teststichprobe
    # die Klassenzugehoerigkeit bekannt ist, kann man am Ende die Klassifikationsergebnisse mit
    # den wahren Klassenlabels (aus der Teststichprobe) vergleichen und eine Fehlerrate angeben. 
     

    # In dem gegebenen Brown Corpus ist keine Aufteilung in Trainings und Testdaten vorgegeben.
    #
    # Waehlen Sie daher die ersten 80% der Dokumente UEBER ALLE KATEGORIEN als Trainingstichprobe
    # und die letzten 20% der Dokumente UEBER ALLE KATEGORIEN als Teststichprobe.
    #
    # Erklaeren Sie, warum Sie die Stichproben ueber alle Kategorien zusammenstellen MUESSEN. 
    #
    # Bitte beachten Sie, dass wir im Rahmen des Fachprojekts keinen Test auf unbekannten Testdaten
    # simulieren. Wir haben ja bereits fuer die Erstellung der Vokabulars (haeufigste Woerter,
    # siehe oben) den kompletten Datensatz verwendet.
    # Stattdessen betrachten wir hier ein so genanntes Validierungsszenario, in dem wir die 
    # Klassifikationsleistung auf dem Brown Corpus optimieren. Die Ergebnisse lassen sich somit
    # nur sehr bedingt auf unbekannte Daten uebertragen.
    #
    # Erstellen Sie nun die NumPy Arrays train_samples, train_labels, test_samples und test_labels,
    # so dass diese mit den estimate und classify Methoden der Klassen im classificaton Modul
    # verwendet werden koennen. Teilen Sie die Daten wie oben angegeben zu 80% in
    # Trainingsdaten und 20% in Testdaten auf. 
    #
    # Hinweis: Vollziehen Sie nach, wie die Klasse CrossValidation im evaluation Modul
    # funktioniert. Wenn Sie moechten, koennen die Klasse zur Aufteilung der Daten
    # verwenden.
    
    raise NotImplementedError('Implement me')
    
    
    # Klassifizieren Sie nun alle Dokumente der Teststichprobe nach dem Prinzip des k-naechste-
    # Nachbarn Klassifikators. Dabei wird die Distanz zwischen dem Merkmalsvektors eines
    # Testbeispiels und allen Merkmalsvektoren aus der Trainingstichprobe berechnet. Das 
    # Klassenlabel des Testbeispiels wird dann ueber einen Mehrheitsentscheid der Klassenlabels 
    # der k aehnlichsten Merkmalsvektoren aus der Trainingsstichprobe bestimmt.
    # http://www5.informatik.uni-erlangen.de/fileadmin/Persons/NiemannHeinrich/klassifikation-von-mustern/m00-www.pdf (Abschnitt 4.2.7)
    #
    # Bestimmen Sie die Distanzen von Testdaten zu Trainingsdaten mit cdist:
    # http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
    # Bestimmen Sie die k-naechsten Nachbarn auf Grundlage der zuvor berechneten
    # Distanzen mit argsort:
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
    # Ueberlegen Sie, welche zuvor von Ihnen implementierte Funktion Sie wiederverwenden
    # koennen, um den Mehrheitsentscheid umzusetzen.
    #
    # Implementieren Sie die Funktionen estimate und classify in der Klasse KNNClassifier 
    # im Modul classification.
    #
    # Verwenden Sie die Euklidische Distanz und betrachten Sie zunaechst nur 
    # den naechsten Nachbarn (k=1).
    #
    # HINWEIS: Hier ist zunaechst nur die Implementierung eines naechster Nachbar 
    # Klassifikators erforderlich. Diese soll aber in der naechsten Aufgabe zu einer 
    # Implementierung eines k-naechste Nachbarn Klassifikators erweitert werden.
    # Beruechsichtigen Sie das in ihrer Implementierung. Um die beiden Varianten
    # zu testen, finden Sie zwei Unit-Test Methoden test_nn und test_knn in dem Modul 
    # test.classification_test
    # 
    
    raise NotImplementedError('Implement me')
    
    # Nachdem Sie mit dem KNNClassifier fuer jedes Testbeispiel ein Klassenlabel geschaetzt haben,
    # koennen Sie dieses mit dem tatsaechlichen Klassenlabel vergleichen. Dieses koennen Sie wie
    # bei den Traingingsdaten dem Corpus entnehmen.
    #
    # Ermitteln Sie eine Gesamtfehlerrate und je eine Fehlerrate pro Kategorie.
    # Implementieren Sie dazu die Klasse ClassificationEvaluator im evaluation Modul. 
    #
    # Warum ist diese Aufteilung der Daten in Training und Test problematisch? Was sagen die Ergebnisse aus?
    
    raise NotImplementedError('Implement me')
    
    
if __name__ == '__main__':
    aufgabe2()
