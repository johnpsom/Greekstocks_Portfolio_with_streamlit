# Greekstocks_Portfolio_with_streamlit [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/johnpsom/greekstocks_portfolio_with_streamlit/main/streamlit_greekstocks.py)


ΠΡΟΣΟΧΗ ότι βλέπετε εδώ είναι φτιαγμένο για ενημερωτικούς και εκπαιδευτικούς σκοπούς μόνο και σε καμιά περίπτωση δεν αποτελεί επενδυτική
ή άλλου είδους πρόταση.Οι επενδύσεις σε μετοχές ενέχουν οικονομικό ρίσκο και ο δημιουργός της εφαρμογής δεν φέρει καμιά ευθύνη σε περίπτωση
απώλειας περιουσίας. Μπορείτε να επικοινωνείτε τα σχόλια και παρατηρήσεις σας στο email: getyour.portfolio@gmail.com .

Υπολογισμός βέλτιστου χαρτοφυλακίου από 100+ επιλεγμένες μετοχές του ΧΑ, βασισμένος στις αρχές της Σύγχρονης Θεωρίας Χαρτοφυλακίου.
Στόχος είναι να αυτοματοποιηθεί όλη η διαδικασία με βάση την στρατηγική που περιγράφεται παρακάτω.

Η εφαρμογή χρησιμοποιεί ιστορικές τιμές για όλες τις μετοχές της παραπάνω λίστας. Οι επιπλεόν παράμετροι της στρατηγικής μας είναι
- ένας τεχνικός δείκτης momentum για να βρίσκει ποιές από αυτές έχουν δυναμική για άνοδο της τιμής τους
- το μέγιστο πλήθος μετοχών που θέλουμε να έχουμε στο χαρτοφυλάκιό μας (π.χ. 5, 10 ή 15 μετοχές)
- το ελάχιστο ποσοστό συμμετοχής της κάθε μετοχής στο επιλεγμένο χαρτοφυλάκιο. (π.χ 5% ή 10%)
- το χρονικό διάστημα διακράτησης του προτεινόμενου χαρτοφυλακίου σε ημέρες (π.χ. 5, 10 ή 20 μέρες)

Η στρατηγική μας είναι αφού δοκιμάσουμε όλους τους συνδυασμούς των παραπάνω παραμέτρων στο παρελθόν με χρήση των ιστορικών τιμών όλων 
των μετοχών να επιλέγουμε κάθε φορά τον καλύτερο συνδυασμό και μετά να δημιουργούμε το χαρτοφυλάκιό μας, ελπίζοντας ότι η δυναμική αυτή θα
είναι σε ισχύ για κάποιο χρονικό διάστημα ακόμη και τουλάχιστον ίσο με το χρονικό διάστημα διακράτησης με βάση το οποίο υπολογίστηκε το 
χαρτοφυλάκιο. 

Κατόπιν θα μπορουμε να το ξανατρέξουμε και να βρεθεί ένας νέος συνδυασμός των παραπάνω παραμέτρων που θα μας δώσει ένα άλλο χαρτοφυλάκιο 
οπότε και θα πρέπει να μεταβάλουμε τις αρχικές μας θέσεις ώστε να γίνουν ίδιες με το νέο.
