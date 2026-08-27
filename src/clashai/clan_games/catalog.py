# clashai/clan_games/catalog.py
# « Puis-je réussir ce défi ? » — logique PURE, aucun modèle, aucun tap.
#
# Tout ce module ne manipule que des chaînes et des dataclasses : il se teste
# intégralement sans jeu, sans GPU et sans émulateur. C'est voulu — c'est ici
# que vivent les décisions, donc c'est ici qu'il faut pouvoir tout couvrir.
#
# ── L'idée centrale ─────────────────────────────────────────────────────────
# L'agent ne JOUE pas les défis : la progression s'incrémente pendant que
# CombatAgent farme. « Puis-je le réussir ? » n'est donc pas un raisonnement,
# c'est un FILTRE DE CAPACITÉS : le défi exige des choses, le bot en possède
# certaines, l'intersection décide.
#
# ── Pourquoi on ne catalogue PAS les défis ──────────────────────────────────
# Première version : un gabarit de phrase par famille de défi. Erreur. Il en
# existe des dizaines (piller de l'or, détruire des murs, gagner des étoiles,
# infliger des dégâts avec les héros…), renouvelés à chaque saison, et tout ce
# qui ne matchait aucun gabarit était refusé — donc la majorité des défis.
#
# On ne décrit donc plus des défis, mais des CONTRAINTES, en deux couches :
#
#   TERRAIN        où ça se joue        village principal / ouvriers / guerre
#   MODIFICATEURS  ce que ça impose     « en utilisant au moins 1 Golem »
#
# **L'objectif lui-même n'entre pas dans la faisabilité.** Que le défi demande
# de piller 500 000 d'or ou de détruire 300 murs ne change rien à « le bot
# peut-il ? » — seulement à l'effort. Un défi jamais vu se lit donc
# correctement du moment que son terrain est reconnu.
#
# Le garde-fou reste : **aucun terrain reconnu = refus**. Une phrase dont on ne
# sait même pas où elle se joue ne peut pas être jugée faisable.
#
# ── « en utilisant au moins 1 Golem » : ce qui bloque vraiment ──────────────
# Ces défis n'exigent pas d'ENTRAÎNER quoi que ce soit de coûteux : depuis la
# refonte du jeu, former des troupes est **instantané et gratuit**. Il n'y a ni
# délai ni dépense à arbitrer, et la limite de 3H d'un défi engagé ne risque
# rien — on peut engager, former, attaquer.
#
# Ce qui bloque est donc une **capacité**, pas un coût : le bot ne sait pas
# encore CHOISIR sa composition. Deux chemins pour qu'un tel défi soit faisable :
#   1. la troupe est DÉJÀ dans la barre (`world['troop_bar']`, capteur existant) ;
#   2. `Capacites.composition_armee` est vrai — le bot sait la former.
# Le jour où (2) existe, basculer le drapeau débloque d'un coup la moitié des
# défis d'une saison (4 sur 7 au relevé du 22 août 2026).
#
# ── Règle de refus ──────────────────────────────────────────────────────────
# Accepter un défi est irréversible (le rejeter a une pénalité en jeu). Donc :
# description non reconnue, unité inconnue du registre, capacité manquante →
# on ne choisit RIEN. Jamais d'approximation, même plausible. Même principe que
# l'anti-gemmes de l'upgrader.

import json
import re
import unicodedata
from dataclasses import dataclass, field

from clashai.paths import CONFIGS_DIR

CATALOGUE_FILE = 'clan_games.json'

# Préfixes de nommage du jeu qui n'existent pas dans le registre : « Sort de
# saut » y est simplement `saut`. Retirés avant la résolution.
_PREFIXES = ('sort de ', 'sorts de ', 'machine de ', 'engin de ')


# =============================================================================
# NORMALISATION
# =============================================================================

def normaliser(texte):
    """Texte OCR -> forme canonique pour le matching des motifs.

    Minuscules, sans accents, espaces compactés, et confusions de chiffres
    corrigées. Cette dernière n'est pas cosmétique : l'OCR rend « Détruisez
    Canon I0 Fois » — sans correction, la quantité est illisible et le défi
    devient inconnu, donc refusé à tort.

    La correction est **bornée aux tokens qui contiennent déjà un chiffre**,
    pour ne pas transformer un mot comme « Golem » ou « Ile ».
    """
    if not texte:
        return ''
    texte = unicodedata.normalize('NFKD', texte)
    texte = ''.join(c for c in texte if not unicodedata.combining(c))
    texte = texte.lower()

    mots = []
    for mot in texte.split():
        if any(c.isdigit() for c in mot):
            mot = mot.replace('i', '1').replace('l', '1').replace('o', '0')
        mots.append(mot)
    return ' '.join(mots)


def _slugifier(nom):
    """« Bébé dragon » -> bebe_dragon ; « Sort de saut » -> saut."""
    nom = normaliser(nom).strip(' .')
    for prefixe in _PREFIXES:
        if nom.startswith(prefixe):
            nom = nom[len(prefixe):]
            break
    return re.sub(r'[^a-z0-9]+', '_', nom).strip('_')


# =============================================================================
# DONNÉES
# =============================================================================

@dataclass
class Analyse:
    """Ce qu'on a compris d'une description. Rien d'inventé : None = pas su."""

    type: str = None                    # id du TERRAIN, None = non reconnu
    contraintes: list = field(default_factory=list)   # ids des modificateurs vus
    quantite: int = None
    unite: str = None                   # slug du registre, None si non résolu
    unite_brute: str = None             # ce que le jeu affichait (diagnostic)
    requires: list = field(default_factory=list)
    effort: int = 3                     # 1 = passif, 3 = hors de portée

    @property
    def reconnu(self) -> bool:
        """Reconnu = on sait OÙ ça se joue. L'objectif n'a pas à l'être."""
        return self.type is not None


@dataclass
class Capacites:
    """Ce que le bot sait faire, ici et maintenant.

    `unites_disponibles` vient de la barre de troupes DU MOMENT — pas de ce que
    le bot pourrait entraîner. C'est la lecture, pas l'intention.
    """

    attaque_multijoueur: bool = True
    village_ouvriers: bool = False
    guerre: bool = False
    unites_disponibles: set = field(default_factory=set)
    # Le bot sait-il CHOISIR ses troupes ? Faux aujourd'hui — l'outil n'existe
    # pas encore. Le jour où il existera, basculer ce drapeau suffit à rendre
    # faisables tous les défis « en utilisant au moins 1 X » dont X est au
    # registre : depuis la refonte du jeu, former des troupes est **instantané
    # et gratuit**, donc il n'y a plus ni délai ni coût à modéliser.
    composition_armee: bool = False


# =============================================================================
# CATALOGUE
# =============================================================================

class Catalogue:
    """Traduit une description de défi en sémantique, puis juge sa faisabilité."""

    def __init__(self, chemin=None, unites_connues=None):
        self._chemin = chemin or f'{CONFIGS_DIR}/{CATALOGUE_FILE}'
        self._data = None
        self._unites = unites_connues       # None -> chargé depuis troops.json

    # ---- chargement --------------------------------------------------------

    def _charger(self):
        if self._data is None:
            with open(self._chemin, encoding='utf-8') as f:
                self._data = json.load(f)
        return self._data

    def _unites_connues(self):
        """Slugs valides = ceux du registre. Autorité unique sur les noms."""
        if self._unites is None:
            from clashai.combat.troop_registry import load_troop_types
            self._unites = {t['name'] for t in load_troop_types()}
        return self._unites

    def terrains(self):
        return self._charger()['terrains']

    def modificateurs(self):
        return self._charger()['modificateurs']

    # ---- interprétation ----------------------------------------------------

    def interpreter(self, description) -> Analyse:
        """Description (brute, telle que l'OCR l'a rendue) -> Analyse.

        Deux passes : le TERRAIN d'abord (premier motif qui matche gagne, d'où
        l'ordre du JSON — du plus spécifique au repli large), puis TOUS les
        modificateurs applicables s'ajoutent par-dessus.

        `re.search` et non `match` : la description est précédée du TITRE du
        pop-up, que l'OCR massacre (« Garderie PQuR bêbês dRagons Gagnez une
        étoile… »). On cherche les motifs où qu'ils soient dans la ligne.
        """
        texte = normaliser(description)
        if not texte:
            return Analyse()

        data = self._charger()

        terrain = next(
            (t for t in data['terrains']
             if any(re.search(m, texte) for m in t['motifs'])),
            None,
        )
        if terrain is None:
            # On ne sait même pas où ça se joue -> jamais choisi.
            return Analyse()

        analyse = Analyse(
            type=terrain['id'],
            requires=list(terrain.get('requires', [])),
            effort=terrain.get('effort', 3),
        )

        for mod in data.get('modificateurs', []):
            trouve = re.search(mod['motif'], texte)
            if not trouve:
                continue
            analyse.contraintes.append(mod['id'])
            analyse.requires.extend(mod.get('requires', []))
            analyse.effort += mod.get('effort_ajoute', 0)

            groupes = mod.get('groupes', {})
            if 'quantite' in groupes:
                try:
                    analyse.quantite = int(trouve.group(groupes['quantite']))
                except (ValueError, IndexError):
                    analyse.quantite = None
            if 'unite' in groupes:
                brute = trouve.group(groupes['unite']).strip()
                analyse.unite_brute = brute
                analyse.unite = self._resoudre_unite(brute, data)

        return analyse

    def _resoudre_unite(self, nom_affiche, data):
        """Nom affiché -> slug du registre, ou None.

        None n'est PAS un échec silencieux : il rend le défi non choisissable.
        Envoyer le bot sur un défi dont on n'a pas identifié l'unité, c'est
        engager un défi irréversible sur une supposition.
        """
        alias = {k: v for k, v in data.get('alias_unites', {}).items()
                 if not k.startswith('_')}
        cle = normaliser(nom_affiche).strip(' .')
        if cle in alias:
            candidat = alias[cle]
        else:
            candidat = _slugifier(nom_affiche)
        return candidat if candidat in self._unites_connues() else None

    # ---- faisabilité -------------------------------------------------------

    def evaluer(self, analyse, capacites) -> tuple:
        """(faisable: bool, raison: str). La raison est toujours renseignée."""
        if not analyse.reconnu:
            return (False, 'description_non_reconnue')

        for besoin in analyse.requires:
            if besoin == 'attaque_multijoueur' and not capacites.attaque_multijoueur:
                return (False, 'attaques_multijoueur_indisponibles')
            if besoin == 'village_ouvriers' and not capacites.village_ouvriers:
                return (False, 'village_des_ouvriers_non_joue')
            if besoin == 'guerre' and not capacites.guerre:
                return (False, 'guerre_non_jouee')
            if besoin == 'unite_disponible':
                if analyse.unite is None:
                    return (False, f'unite_non_identifiee:{analyse.unite_brute}')
                if analyse.unite in capacites.unites_disponibles:
                    continue
                # La troupe n'est pas dans la barre. Ce qui bloque n'est PAS un
                # coût — former des troupes est instantané et gratuit depuis la
                # refonte du jeu — mais une CAPACITÉ que le bot n'a pas encore :
                # choisir sa composition. Les deux raisons se distinguent, parce
                # qu'elles appellent des réponses différentes.
                if capacites.composition_armee:
                    continue          # on sait la former -> faisable
                return (False, f'compo_armee_non_pilotee:{analyse.unite}')

        return (True, 'ok')

    # ---- choix -------------------------------------------------------------

    def choisir(self, defis, capacites, interpreter=None):
        """Le meilleur défi engageable, ou None.

        `defis` : itérable de `clan_games.reader.Defi`. Un défi sans points lus
        est écarté — on ne compare pas ce qu'on n'a pas lu.

        Score = points / effort. À égalité, les points bruts tranchent : à
        rendement égal, autant prendre celui qui rapporte le plus d'un coup
        (moins d'allers-retours dans le menu).

        Rend None dès qu'il n'y a aucun candidat SÛR — l'agent réessaiera au
        prochain cooldown plutôt que d'engager un défi irréversible au hasard.
        """
        interpreter = interpreter or self.interpreter

        meilleur, meilleur_score = None, None
        for defi in defis:
            if not defi.disponible or defi.points is None:
                continue
            analyse = interpreter(getattr(defi, 'description', None))
            faisable, _raison = self.evaluer(analyse, capacites)
            if not faisable:
                continue
            score = defi.points / max(1, analyse.effort)
            cle = (score, defi.points)
            if meilleur_score is None or cle > meilleur_score:
                meilleur, meilleur_score = (defi, analyse), cle

        return meilleur

    def defis_a_une_troupe_pres(self, defis, capacites):
        """Défis refusés UNIQUEMENT parce qu'une troupe manque à l'armée.

        C'est un **besoin**, pas une action : l'agent ne compose pas l'armée, il
        signale ce qui la débloquerait. Le cerveau arbitre — adapter la compo
        pour 150 points a un coût (temps, élixir) qu'un défi sans contrainte à
        300 points ne demande pas.

        ⚠️ On sépare l'actionnable du définitif. `village_des_ouvriers_non_joue`
        ne sera JAMAIS débloqué par une troupe : le faire remonter comme un
        besoin enverrait le cerveau sur une fausse piste.

        📌 **Ce n'est pas une question de coût.** Depuis la refonte du jeu, former
        des troupes est **instantané et gratuit** : il n'y a ni délai ni dépense
        à arbitrer, et la limite de 3H d'un défi engagé ne risque rien. Ce qui
        bloque est une **capacité manquante** — le bot ne choisit pas encore sa
        composition. D'où la raison dédiée `compo_armee_non_pilotee`.

        Renvoie [{points, unite, terrain, description}], meilleurs points d'abord.
        """
        out = []
        for defi, analyse, faisable, raison in self.diagnostiquer(defis, capacites):
            if faisable or not raison.startswith('compo_armee_non_pilotee:'):
                continue
            out.append({
                'points': defi.points,
                'unite': analyse.unite,
                'terrain': analyse.type,
                'description': getattr(defi, 'description', None),
            })
        out.sort(key=lambda d: -(d['points'] or 0))
        return out

    def diagnostiquer(self, defis, capacites):
        """[(defi, analyse, faisable, raison)] — pour les démos et les logs."""
        out = []
        for defi in defis:
            analyse = self.interpreter(getattr(defi, 'description', None))
            faisable, raison = self.evaluer(analyse, capacites)
            if defi.points is None and faisable:
                faisable, raison = False, 'points_non_lus'
            elif not defi.disponible and faisable:
                faisable, raison = False, 'carte_grisee_ou_active'
            out.append((defi, analyse, faisable, raison))
        return out


def capacites_depuis_world(world, **surcharges):
    """Construit les `Capacites` depuis le `world` partagé des agents.

    Les unités viennent de `world['troop_bar']` : ce que l'armée contient
    RÉELLEMENT à cet instant. Une barre non lue donne un ensemble vide, donc
    aucun défi « avec unité imposée » — on ne suppose pas qu'une troupe est là.
    """
    unites = set()
    for d in (world or {}).get('troop_bar') or []:
        nom = d.get('name') if isinstance(d, dict) else None
        if nom and not (isinstance(d, dict) and d.get('is_grayed')):
            unites.add(nom)

    caps = Capacites(unites_disponibles=unites)
    for k, v in surcharges.items():
        setattr(caps, k, v)
    return caps
