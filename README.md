## StanDepPy

The proposed MatLab workflow from StanDep paper in Python 

#### AI suggestion

standep_py/
├── standep_py/        # Quellcode
│   ├── __init__.py
│   ├── data.py        # Datenstrukturen & Loader
│   ├── modeldata.py   # getModelData & Co.
│   ├── enzymes.py     # getSpecialistEnzymes, getPromEnzymes, …
│   ├── clustering.py  # geneExprDist_hierarchy
│   ├── core_rxns.py   # models4mClusters1
│   ├── similarity.py  # calcJaccardSimilarity
│   ├── ubiquity.py    # getUbiquityScore
│   └── mCADRE.py      # Wrapper/Interface zu mCADRE
├── tests/             # Unit- und Integrationstests (pytest)
├── examples/          # Jupyter-Notebooks mit Demo-Workflows
├── setup.py / pyproject.toml
└── README.md






project/
├── data/
│   ├── expression_data.mat    # oder .csv, etc.
│   └── model.sbml
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Laden und Konvertieren von Input-Dateien in Dictionaries
│   ├── model_data.py          # Funktion: get_model_data(expression_data, model)
│   ├── gpr_parser.py          # Funktionen: parse_gpr(model), linearization_index(), 
│   │                        #              get_specialist_enzymes(model), get_promiscuous_enzymes(model)
│   ├── enzyme_expression.py   # Funktion: compare_promiscuous_specific(spec, prom, model_data)
│   └── main.py                # Koordiniert den gesamten Workflow
├── requirements.txt
└── README.md

Funktionen im Einzelnen:

    data_loader.py:

        load_expression_data(file_path): Liest verschiedene Formate (.mat, .csv) ein und wandelt sie in ein Dictionary um.

        load_model(file_path): Lädt das Modell (z. B. mit cobra.io.read_sbml_model()).

    model_data.py:

        get_model_data(expression_data, model):

            Konvertiert ggf. numerische Gene-IDs in Strings.

            Filtert die Expression-Daten anhand der Modell-Gene.

            Aggregiert doppelte Einträge (z. B. durch Mittelwertbildung).

            Gibt ein strukturiertes Dictionary mit Feldern wie gene, value, Tissue etc. zurück.

    gpr_parser.py:

        parse_gpr(model): Parst die GPR-Regeln des Modells (analog zu GPRparser()).

        linearization_index(parsed_gpr, mode): Führt eine Linearisierung der GPR-Daten durch.

        get_specialist_enzymes(model): Baut auf parse_gpr() auf, filtert und gibt ein Dictionary mit spezialisierten Enzymen, zugehörigen Reaktionen und Subsystemen zurück.

        get_promiscuous_enzymes(model): Ähnlich wie get_specialist_enzymes(), aber für promiskuitive Enzyme.

    enzyme_expression.py:

        compare_promiscuous_specific(spec, prom, model_data):

            Berechnet Expression-Matrizen für Spezialisten- und Promiscuous-Enzyme (z. B. mittels Minimumsbildung über Subunits).

            Erstellt ggf. Histogramme (mit matplotlib) zur Visualisierung der log10-transformierten Expression.

            Gibt ein Dictionary (enzymeData) zurück, das Enzymnamen, Expression, zugehörige Reaktionen und Tissue-Daten enthält.

    main.py:

        Koordiniert den Ablauf:

            Laden der Expression-Daten und des Modells.

            Erzeugen der model_data mittels get_model_data()

            Parsen der GPR-Regeln und Erzeugen der Enzym-Dictionaries (get_specialist_enzymes() und get_promiscuous_enzymes()).

            Vergleich der Expression (über compare_promiscuous_specific()).

            Ausgabe/Visualisierung der Ergebnisse.

##########################

Hier ein Vorschlag für eine modulare Bibliotheksstruktur, die robust und vielseitig einsetzbar ist:

    Hauptpaket (z. B. enzyme_analysis/):

        __init__.py
        – Exporte der wichtigsten Funktionen für den Nutzer.

        data_loader.py
        – Funktionen:

            load_expression_data(filepath): Liest verschiedene Formate (z. B. .mat, .csv) und wandelt sie in ein Dictionary um.

            load_model(filepath): Lädt das Modell (z. B. via cobra.io.read_sbml_model()).

        model_data.py
        – Funktion:

            get_model_data(expression_data, model): Repliziert die MATLAB-Logik (u. a. Gene-Konvertierung, Filtern, Aggregieren) und liefert ein strukturiertes Dictionary zurück.

        gpr_parser.py
        – Funktionen:

            parse_gpr(model): Parst die GPR-Regeln des Modells.

            linearization_index(parsed_gpr, mode): Führt eine Linearisierung der GPR-Daten durch.

            get_specialist_enzymes(model): Baut auf den GPR-Funktionen auf und extrahiert Spezialisten-Enzyme.

            get_promiscuous_enzymes(model): Ähnlich, aber für promiskuitive Enzyme.

        enzyme_expression.py
        – Funktion:

            compare_promiscuous_specific(spec, prom, model_data): Berechnet die Enzym-Expression (z. B. mittels Minimumsbildung über Subunits), erstellt Visualisierungen und liefert ein strukturiertes Ergebnis (enzymeData).

        utils.py (optional):
        – Gemeinsame Hilfsfunktionen wie Logging, Fehlerbehandlung oder Datenkonvertierung.

    Beispiel-Skripte (examples/):

        example_workflow.py
        – Demonstriert, wie die Bibliotheksfunktionen zusammenspielen, um aus Transcriptomics-Daten die Enzym-Expression zu berechnen.

    Tests (tests/):

        Unit-Tests für die einzelnen Module und Funktionen, um Robustheit und Korrektheit sicherzustellen.

    Distributionsdateien:

        setup.py und requirements.txt zur Installation und Verwaltung von Abhängigkeiten.

Diese Struktur stellt sicher, dass du einen robusten und wiederverwendbaren Workflow als Bibliothek hast – die einzelnen Module bieten klar abgegrenzte Funktionalitäten, während Beispielskripte den praktischen Einsatz demonstrieren.




### Überlegung expression

Möglich expressionsfälle:
- Garnicht expremiert                                   keine kurve                                         alle inaktiv
- Alle gleich                                           normalverteilung mit kleiner standardabweichung     ab kleinem GL active? 
- Verschiedenee cluster mit gleichen expressionen       mehrere normalverteilungen (zu wenig cluster?)      ab einem bestimmten lvl active
- arbiträr                                              kein muster                                         ab einem bestimmten lvl active