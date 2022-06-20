# Copyright (c) Facebook, Inc. and its affiliates.
# Autogen with
# with open("lvis_v0.5_val.json", "r") as f:
#     a = json.load(f)
# c = a["categories"]
# for x in c:
#     del x["image_count"]
#     del x["instance_count"]
# LVIS_CATEGORIES = repr(c) + "  # noqa"

# fmt: off
LVIS_CATEGORIES = [{
    'frequency': 'r',
    'id': 1,
    'synset': 'acorn.n.01',
    'synonyms': ['acorn'],
    'def': 'nut from an oak tree',
    'name': 'acorn'
}, {
    'frequency': 'c',
    'id': 2,
    'synset': 'aerosol.n.02',
    'synonyms': ['aerosol_can', 'spray_can'],
    'def': 'a dispenser that holds a substance under pressure',
    'name': 'aerosol_can'
}, {
    'frequency': 'f',
    'id': 3,
    'synset': 'air_conditioner.n.01',
    'synonyms': ['air_conditioner'],
    'def': 'a machine that keeps air cool and dry',
    'name': 'air_conditioner'
}, {
    'frequency':
        'f',
    'id':
        4,
    'synset':
        'airplane.n.01',
    'synonyms': ['airplane', 'aeroplane'],
    'def':
        'an aircraft that has a fixed wing and is powered by propellers or jets',
    'name':
        'airplane'
}, {
    'frequency': 'c',
    'id': 5,
    'synset': 'alarm_clock.n.01',
    'synonyms': ['alarm_clock'],
    'def': 'a clock that wakes a sleeper at some preset time',
    'name': 'alarm_clock'
}, {
    'frequency': 'c',
    'id': 6,
    'synset': 'alcohol.n.01',
    'synonyms': ['alcohol', 'alcoholic_beverage'],
    'def': 'a liquor or brew containing alcohol as the active agent',
    'name': 'alcohol'
}, {
    'frequency':
        'r',
    'id':
        7,
    'synset':
        'alligator.n.02',
    'synonyms': ['alligator', 'gator'],
    'def':
        'amphibious reptiles related to crocodiles but with shorter broader snouts',
    'name':
        'alligator'
}, {
    'frequency': 'c',
    'id': 8,
    'synset': 'almond.n.02',
    'synonyms': ['almond'],
    'def': 'oval-shaped edible seed of the almond tree',
    'name': 'almond'
}, {
    'frequency': 'c',
    'id': 9,
    'synset': 'ambulance.n.01',
    'synonyms': ['ambulance'],
    'def': 'a vehicle that takes people to and from hospitals',
    'name': 'ambulance'
}, {
    'frequency': 'r',
    'id': 10,
    'synset': 'amplifier.n.01',
    'synonyms': ['amplifier'],
    'def': 'electronic equipment that increases strength of signals',
    'name': 'amplifier'
}, {
    'frequency': 'c',
    'id': 11,
    'synset': 'anklet.n.03',
    'synonyms': ['anklet', 'ankle_bracelet'],
    'def': 'an ornament worn around the ankle',
    'name': 'anklet'
}, {
    'frequency':
        'f',
    'id':
        12,
    'synset':
        'antenna.n.01',
    'synonyms': ['antenna', 'aerial', 'transmitting_aerial'],
    'def':
        'an electrical device that sends or receives radio or television signals',
    'name':
        'antenna'
}, {
    'frequency':
        'f',
    'id':
        13,
    'synset':
        'apple.n.01',
    'synonyms': ['apple'],
    'def':
        'fruit with red or yellow or green skin and sweet to tart crisp whitish flesh',
    'name':
        'apple'
}, {
    'frequency': 'r',
    'id': 14,
    'synset': 'apple_juice.n.01',
    'synonyms': ['apple_juice'],
    'def': 'the juice of apples',
    'name': 'apple_juice'
}, {
    'frequency': 'r',
    'id': 15,
    'synset': 'applesauce.n.01',
    'synonyms': ['applesauce'],
    'def': 'puree of stewed apples usually sweetened and spiced',
    'name': 'applesauce'
}, {
    'frequency': 'r',
    'id': 16,
    'synset': 'apricot.n.02',
    'synonyms': ['apricot'],
    'def': 'downy yellow to rosy-colored fruit resembling a small peach',
    'name': 'apricot'
}, {
    'frequency':
        'f',
    'id':
        17,
    'synset':
        'apron.n.01',
    'synonyms': ['apron'],
    'def':
        'a garment of cloth that is tied about the waist and worn to protect clothing',
    'name':
        'apron'
}, {
    'frequency':
        'c',
    'id':
        18,
    'synset':
        'aquarium.n.01',
    'synonyms': ['aquarium', 'fish_tank'],
    'def':
        'a tank/pool/bowl filled with water for keeping live fish and underwater animals',
    'name':
        'aquarium'
}, {
    'frequency': 'c',
    'id': 19,
    'synset': 'armband.n.02',
    'synonyms': ['armband'],
    'def': 'a band worn around the upper arm',
    'name': 'armband'
}, {
    'frequency': 'f',
    'id': 20,
    'synset': 'armchair.n.01',
    'synonyms': ['armchair'],
    'def': 'chair with a support on each side for arms',
    'name': 'armchair'
}, {
    'frequency': 'r',
    'id': 21,
    'synset': 'armoire.n.01',
    'synonyms': ['armoire'],
    'def': 'a large wardrobe or cabinet',
    'name': 'armoire'
}, {
    'frequency': 'r',
    'id': 22,
    'synset': 'armor.n.01',
    'synonyms': ['armor', 'armour'],
    'def': 'protective covering made of metal and used in combat',
    'name': 'armor'
}, {
    'frequency': 'c',
    'id': 23,
    'synset': 'artichoke.n.02',
    'synonyms': ['artichoke'],
    'def': 'a thistlelike flower head with edible fleshy leaves and heart',
    'name': 'artichoke'
}, {
    'frequency': 'f',
    'id': 24,
    'synset': 'ashcan.n.01',
    'synonyms': [
        'trash_can', 'garbage_can', 'wastebin', 'dustbin', 'trash_barrel',
        'trash_bin'
    ],
    'def': 'a bin that holds rubbish until it is collected',
    'name': 'trash_can'
}, {
    'frequency': 'c',
    'id': 25,
    'synset': 'ashtray.n.01',
    'synonyms': ['ashtray'],
    'def': "a receptacle for the ash from smokers' cigars or cigarettes",
    'name': 'ashtray'
}, {
    'frequency': 'c',
    'id': 26,
    'synset': 'asparagus.n.02',
    'synonyms': ['asparagus'],
    'def': 'edible young shoots of the asparagus plant',
    'name': 'asparagus'
}, {
    'frequency': 'c',
    'id': 27,
    'synset': 'atomizer.n.01',
    'synonyms': [
        'atomizer', 'atomiser', 'spray', 'sprayer', 'nebulizer', 'nebuliser'
    ],
    'def': 'a dispenser that turns a liquid (such as perfume) into a fine mist',
    'name': 'atomizer'
}, {
    'frequency':
        'c',
    'id':
        28,
    'synset':
        'avocado.n.01',
    'synonyms': ['avocado'],
    'def':
        'a pear-shaped fruit with green or blackish skin and rich yellowish pulp enclosing a single large seed',
    'name':
        'avocado'
}, {
    'frequency': 'c',
    'id': 29,
    'synset': 'award.n.02',
    'synonyms': ['award', 'accolade'],
    'def': 'a tangible symbol signifying approval or distinction',
    'name': 'award'
}, {
    'frequency':
        'f',
    'id':
        30,
    'synset':
        'awning.n.01',
    'synonyms': ['awning'],
    'def':
        'a canopy made of canvas to shelter people or things from rain or sun',
    'name':
        'awning'
}, {
    'frequency': 'r',
    'id': 31,
    'synset': 'ax.n.01',
    'synonyms': ['ax', 'axe'],
    'def': 'an edge tool with a heavy bladed head mounted across a handle',
    'name': 'ax'
}, {
    'frequency':
        'f',
    'id':
        32,
    'synset':
        'baby_buggy.n.01',
    'synonyms': [
        'baby_buggy', 'baby_carriage', 'perambulator', 'pram', 'stroller'
    ],
    'def':
        'a small vehicle with four wheels in which a baby or child is pushed around',
    'name':
        'baby_buggy'
}, {
    'frequency':
        'c',
    'id':
        33,
    'synset':
        'backboard.n.01',
    'synonyms': ['basketball_backboard'],
    'def':
        'a raised vertical board with basket attached; used to play basketball',
    'name':
        'basketball_backboard'
}, {
    'frequency': 'f',
    'id': 34,
    'synset': 'backpack.n.01',
    'synonyms': ['backpack', 'knapsack', 'packsack', 'rucksack', 'haversack'],
    'def': 'a bag carried by a strap on your back or shoulder',
    'name': 'backpack'
}, {
    'frequency':
        'f',
    'id':
        35,
    'synset':
        'bag.n.04',
    'synonyms': ['handbag', 'purse', 'pocketbook'],
    'def':
        'a container used for carrying money and small personal items or accessories',
    'name':
        'handbag'
}, {
    'frequency': 'f',
    'id': 36,
    'synset': 'bag.n.06',
    'synonyms': ['suitcase', 'baggage', 'luggage'],
    'def': 'cases used to carry belongings when traveling',
    'name': 'suitcase'
}, {
    'frequency': 'c',
    'id': 37,
    'synset': 'bagel.n.01',
    'synonyms': ['bagel', 'beigel'],
    'def': 'glazed yeast-raised doughnut-shaped roll with hard crust',
    'name': 'bagel'
}, {
    'frequency':
        'r',
    'id':
        38,
    'synset':
        'bagpipe.n.01',
    'synonyms': ['bagpipe'],
    'def':
        'a tubular wind instrument; the player blows air into a bag and squeezes it out',
    'name':
        'bagpipe'
}, {
    'frequency': 'r',
    'id': 39,
    'synset': 'baguet.n.01',
    'synonyms': ['baguet', 'baguette'],
    'def': 'narrow French stick loaf',
    'name': 'baguet'
}, {
    'frequency':
        'r',
    'id':
        40,
    'synset':
        'bait.n.02',
    'synonyms': ['bait', 'lure'],
    'def':
        'something used to lure fish or other animals into danger so they can be trapped or killed',
    'name':
        'bait'
}, {
    'frequency': 'f',
    'id': 41,
    'synset': 'ball.n.06',
    'synonyms': ['ball'],
    'def': 'a spherical object used as a plaything',
    'name': 'ball'
}, {
    'frequency': 'r',
    'id': 42,
    'synset': 'ballet_skirt.n.01',
    'synonyms': ['ballet_skirt', 'tutu'],
    'def': 'very short skirt worn by ballerinas',
    'name': 'ballet_skirt'
}, {
    'frequency': 'f',
    'id': 43,
    'synset': 'balloon.n.01',
    'synonyms': ['balloon'],
    'def': 'large tough nonrigid bag filled with gas or heated air',
    'name': 'balloon'
}, {
    'frequency': 'c',
    'id': 44,
    'synset': 'bamboo.n.02',
    'synonyms': ['bamboo'],
    'def': 'woody tropical grass having hollow woody stems',
    'name': 'bamboo'
}, {
    'frequency': 'f',
    'id': 45,
    'synset': 'banana.n.02',
    'synonyms': ['banana'],
    'def': 'elongated crescent-shaped yellow fruit with soft sweet flesh',
    'name': 'banana'
}, {
    'frequency': 'r',
    'id': 46,
    'synset': 'band_aid.n.01',
    'synonyms': ['Band_Aid'],
    'def': 'trade name for an adhesive bandage to cover small cuts or blisters',
    'name': 'Band_Aid'
}, {
    'frequency':
        'c',
    'id':
        47,
    'synset':
        'bandage.n.01',
    'synonyms': ['bandage'],
    'def':
        'a piece of soft material that covers and protects an injured part of the body',
    'name':
        'bandage'
}, {
    'frequency':
        'c',
    'id':
        48,
    'synset':
        'bandanna.n.01',
    'synonyms': ['bandanna', 'bandana'],
    'def':
        'large and brightly colored handkerchief; often used as a neckerchief',
    'name':
        'bandanna'
}, {
    'frequency':
        'r',
    'id':
        49,
    'synset':
        'banjo.n.01',
    'synonyms': ['banjo'],
    'def':
        'a stringed instrument of the guitar family with a long neck and circular body',
    'name':
        'banjo'
}, {
    'frequency': 'f',
    'id': 50,
    'synset': 'banner.n.01',
    'synonyms': ['banner', 'streamer'],
    'def': 'long strip of cloth or paper used for decoration or advertising',
    'name': 'banner'
}, {
    'frequency':
        'r',
    'id':
        51,
    'synset':
        'barbell.n.01',
    'synonyms': ['barbell'],
    'def':
        'a bar to which heavy discs are attached at each end; used in weightlifting',
    'name':
        'barbell'
}, {
    'frequency': 'r',
    'id': 52,
    'synset': 'barge.n.01',
    'synonyms': ['barge'],
    'def': 'a flatbottom boat for carrying heavy loads (especially on canals)',
    'name': 'barge'
}, {
    'frequency': 'f',
    'id': 53,
    'synset': 'barrel.n.02',
    'synonyms': ['barrel', 'cask'],
    'def': 'a cylindrical container that holds liquids',
    'name': 'barrel'
}, {
    'frequency': 'c',
    'id': 54,
    'synset': 'barrette.n.01',
    'synonyms': ['barrette'],
    'def': "a pin for holding women's hair in place",
    'name': 'barrette'
}, {
    'frequency':
        'c',
    'id':
        55,
    'synset':
        'barrow.n.03',
    'synonyms': ['barrow', 'garden_cart', 'lawn_cart', 'wheelbarrow'],
    'def':
        'a cart for carrying small loads; has handles and one or more wheels',
    'name':
        'barrow'
}, {
    'frequency': 'f',
    'id': 56,
    'synset': 'base.n.03',
    'synonyms': ['baseball_base'],
    'def': 'a place that the runner must touch before scoring',
    'name': 'baseball_base'
}, {
    'frequency': 'f',
    'id': 57,
    'synset': 'baseball.n.02',
    'synonyms': ['baseball'],
    'def': 'a ball used in playing baseball',
    'name': 'baseball'
}, {
    'frequency': 'f',
    'id': 58,
    'synset': 'baseball_bat.n.01',
    'synonyms': ['baseball_bat'],
    'def': 'an implement used in baseball by the batter',
    'name': 'baseball_bat'
}, {
    'frequency': 'f',
    'id': 59,
    'synset': 'baseball_cap.n.01',
    'synonyms': ['baseball_cap', 'jockey_cap', 'golf_cap'],
    'def': 'a cap with a bill',
    'name': 'baseball_cap'
}, {
    'frequency': 'f',
    'id': 60,
    'synset': 'baseball_glove.n.01',
    'synonyms': ['baseball_glove', 'baseball_mitt'],
    'def': 'the handwear used by fielders in playing baseball',
    'name': 'baseball_glove'
}, {
    'frequency': 'f',
    'id': 61,
    'synset': 'basket.n.01',
    'synonyms': ['basket', 'handbasket'],
    'def': 'a container that is usually woven and has handles',
    'name': 'basket'
}, {
    'frequency':
        'c',
    'id':
        62,
    'synset':
        'basket.n.03',
    'synonyms': ['basketball_hoop'],
    'def':
        'metal hoop supporting a net through which players try to throw the basketball',
    'name':
        'basketball_hoop'
}, {
    'frequency': 'c',
    'id': 63,
    'synset': 'basketball.n.02',
    'synonyms': ['basketball'],
    'def': 'an inflated ball used in playing basketball',
    'name': 'basketball'
}, {
    'frequency': 'r',
    'id': 64,
    'synset': 'bass_horn.n.01',
    'synonyms': ['bass_horn', 'sousaphone', 'tuba'],
    'def': 'the lowest brass wind instrument',
    'name': 'bass_horn'
}, {
    'frequency':
        'r',
    'id':
        65,
    'synset':
        'bat.n.01',
    'synonyms': ['bat_(animal)'],
    'def':
        'nocturnal mouselike mammal with forelimbs modified to form membranous wings',
    'name':
        'bat_(animal)'
}, {
    'frequency':
        'f',
    'id':
        66,
    'synset':
        'bath_mat.n.01',
    'synonyms': ['bath_mat'],
    'def':
        'a heavy towel or mat to stand on while drying yourself after a bath',
    'name':
        'bath_mat'
}, {
    'frequency': 'f',
    'id': 67,
    'synset': 'bath_towel.n.01',
    'synonyms': ['bath_towel'],
    'def': 'a large towel; to dry yourself after a bath',
    'name': 'bath_towel'
}, {
    'frequency': 'c',
    'id': 68,
    'synset': 'bathrobe.n.01',
    'synonyms': ['bathrobe'],
    'def': 'a loose-fitting robe of towelling; worn after a bath or swim',
    'name': 'bathrobe'
}, {
    'frequency':
        'f',
    'id':
        69,
    'synset':
        'bathtub.n.01',
    'synonyms': ['bathtub', 'bathing_tub'],
    'def':
        'a large open container that you fill with water and use to wash the body',
    'name':
        'bathtub'
}, {
    'frequency':
        'r',
    'id':
        70,
    'synset':
        'batter.n.02',
    'synonyms': ['batter_(food)'],
    'def':
        'a liquid or semiliquid mixture, as of flour, eggs, and milk, used in cooking',
    'name':
        'batter_(food)'
}, {
    'frequency': 'c',
    'id': 71,
    'synset': 'battery.n.02',
    'synonyms': ['battery'],
    'def': 'a portable device that produces electricity',
    'name': 'battery'
}, {
    'frequency': 'r',
    'id': 72,
    'synset': 'beach_ball.n.01',
    'synonyms': ['beachball'],
    'def': 'large and light ball; for play at the seaside',
    'name': 'beachball'
}, {
    'frequency':
        'c',
    'id':
        73,
    'synset':
        'bead.n.01',
    'synonyms': ['bead'],
    'def':
        'a small ball with a hole through the middle used for ornamentation, jewellery, etc.',
    'name':
        'bead'
}, {
    'frequency': 'r',
    'id': 74,
    'synset': 'beaker.n.01',
    'synonyms': ['beaker'],
    'def': 'a flatbottomed jar made of glass or plastic; used for chemistry',
    'name': 'beaker'
}, {
    'frequency': 'c',
    'id': 75,
    'synset': 'bean_curd.n.01',
    'synonyms': ['bean_curd', 'tofu'],
    'def': 'cheeselike food made of curdled soybean milk',
    'name': 'bean_curd'
}, {
    'frequency':
        'c',
    'id':
        76,
    'synset':
        'beanbag.n.01',
    'synonyms': ['beanbag'],
    'def':
        'a bag filled with dried beans or similar items; used in games or to sit on',
    'name':
        'beanbag'
}, {
    'frequency': 'f',
    'id': 77,
    'synset': 'beanie.n.01',
    'synonyms': ['beanie', 'beany'],
    'def': 'a small skullcap; formerly worn by schoolboys and college freshmen',
    'name': 'beanie'
}, {
    'frequency':
        'f',
    'id':
        78,
    'synset':
        'bear.n.01',
    'synonyms': ['bear'],
    'def':
        'large carnivorous or omnivorous mammals with shaggy coats and claws',
    'name':
        'bear'
}, {
    'frequency': 'f',
    'id': 79,
    'synset': 'bed.n.01',
    'synonyms': ['bed'],
    'def': 'a piece of furniture that provides a place to sleep',
    'name': 'bed'
}, {
    'frequency': 'c',
    'id': 80,
    'synset': 'bedspread.n.01',
    'synonyms': [
        'bedspread', 'bedcover', 'bed_covering', 'counterpane', 'spread'
    ],
    'def': 'decorative cover for a bed',
    'name': 'bedspread'
}, {
    'frequency': 'f',
    'id': 81,
    'synset': 'beef.n.01',
    'synonyms': ['cow'],
    'def': 'cattle that are reared for their meat',
    'name': 'cow'
}, {
    'frequency': 'c',
    'id': 82,
    'synset': 'beef.n.02',
    'synonyms': ['beef_(food)', 'boeuf_(food)'],
    'def': 'meat from an adult domestic bovine',
    'name': 'beef_(food)'
}, {
    'frequency': 'r',
    'id': 83,
    'synset': 'beeper.n.01',
    'synonyms': ['beeper', 'pager'],
    'def': 'an device that beeps when the person carrying it is being paged',
    'name': 'beeper'
}, {
    'frequency': 'f',
    'id': 84,
    'synset': 'beer_bottle.n.01',
    'synonyms': ['beer_bottle'],
    'def': 'a bottle that holds beer',
    'name': 'beer_bottle'
}, {
    'frequency': 'c',
    'id': 85,
    'synset': 'beer_can.n.01',
    'synonyms': ['beer_can'],
    'def': 'a can that holds beer',
    'name': 'beer_can'
}, {
    'frequency': 'r',
    'id': 86,
    'synset': 'beetle.n.01',
    'synonyms': ['beetle'],
    'def': 'insect with hard wing covers',
    'name': 'beetle'
}, {
    'frequency':
        'f',
    'id':
        87,
    'synset':
        'bell.n.01',
    'synonyms': ['bell'],
    'def':
        'a hollow device made of metal that makes a ringing sound when struck',
    'name':
        'bell'
}, {
    'frequency':
        'f',
    'id':
        88,
    'synset':
        'bell_pepper.n.02',
    'synonyms': ['bell_pepper', 'capsicum'],
    'def':
        'large bell-shaped sweet pepper in green or red or yellow or orange or black varieties',
    'name':
        'bell_pepper'
}, {
    'frequency': 'f',
    'id': 89,
    'synset': 'belt.n.02',
    'synonyms': ['belt'],
    'def': 'a band to tie or buckle around the body (usually at the waist)',
    'name': 'belt'
}, {
    'frequency': 'f',
    'id': 90,
    'synset': 'belt_buckle.n.01',
    'synonyms': ['belt_buckle'],
    'def': 'the buckle used to fasten a belt',
    'name': 'belt_buckle'
}, {
    'frequency': 'f',
    'id': 91,
    'synset': 'bench.n.01',
    'synonyms': ['bench'],
    'def': 'a long seat for more than one person',
    'name': 'bench'
}, {
    'frequency': 'c',
    'id': 92,
    'synset': 'beret.n.01',
    'synonyms': ['beret'],
    'def': 'a cap with no brim or bill; made of soft cloth',
    'name': 'beret'
}, {
    'frequency': 'c',
    'id': 93,
    'synset': 'bib.n.02',
    'synonyms': ['bib'],
    'def': 'a napkin tied under the chin of a child while eating',
    'name': 'bib'
}, {
    'frequency': 'r',
    'id': 94,
    'synset': 'bible.n.01',
    'synonyms': ['Bible'],
    'def': 'the sacred writings of the Christian religions',
    'name': 'Bible'
}, {
    'frequency': 'f',
    'id': 95,
    'synset': 'bicycle.n.01',
    'synonyms': ['bicycle', 'bike_(bicycle)'],
    'def': 'a wheeled vehicle that has two wheels and is moved by foot pedals',
    'name': 'bicycle'
}, {
    'frequency': 'f',
    'id': 96,
    'synset': 'bill.n.09',
    'synonyms': ['visor', 'vizor'],
    'def': 'a brim that projects to the front to shade the eyes',
    'name': 'visor'
}, {
    'frequency': 'c',
    'id': 97,
    'synset': 'binder.n.03',
    'synonyms': ['binder', 'ring-binder'],
    'def': 'holds loose papers or magazines',
    'name': 'binder'
}, {
    'frequency': 'c',
    'id': 98,
    'synset': 'binoculars.n.01',
    'synonyms': ['binoculars', 'field_glasses', 'opera_glasses'],
    'def': 'an optical instrument designed for simultaneous use by both eyes',
    'name': 'binoculars'
}, {
    'frequency': 'f',
    'id': 99,
    'synset': 'bird.n.01',
    'synonyms': ['bird'],
    'def': 'animal characterized by feathers and wings',
    'name': 'bird'
}, {
    'frequency': 'r',
    'id': 100,
    'synset': 'bird_feeder.n.01',
    'synonyms': ['birdfeeder'],
    'def': 'an outdoor device that supplies food for wild birds',
    'name': 'birdfeeder'
}, {
    'frequency': 'r',
    'id': 101,
    'synset': 'birdbath.n.01',
    'synonyms': ['birdbath'],
    'def': 'an ornamental basin (usually in a garden) for birds to bathe in',
    'name': 'birdbath'
}, {
    'frequency': 'c',
    'id': 102,
    'synset': 'birdcage.n.01',
    'synonyms': ['birdcage'],
    'def': 'a cage in which a bird can be kept',
    'name': 'birdcage'
}, {
    'frequency': 'c',
    'id': 103,
    'synset': 'birdhouse.n.01',
    'synonyms': ['birdhouse'],
    'def': 'a shelter for birds',
    'name': 'birdhouse'
}, {
    'frequency': 'f',
    'id': 104,
    'synset': 'birthday_cake.n.01',
    'synonyms': ['birthday_cake'],
    'def': 'decorated cake served at a birthday party',
    'name': 'birthday_cake'
}, {
    'frequency': 'r',
    'id': 105,
    'synset': 'birthday_card.n.01',
    'synonyms': ['birthday_card'],
    'def': 'a card expressing a birthday greeting',
    'name': 'birthday_card'
}, {
    'frequency': 'r',
    'id': 106,
    'synset': 'biscuit.n.01',
    'synonyms': ['biscuit_(bread)'],
    'def': 'small round bread leavened with baking-powder or soda',
    'name': 'biscuit_(bread)'
}, {
    'frequency':
        'r',
    'id':
        107,
    'synset':
        'black_flag.n.01',
    'synonyms': ['pirate_flag'],
    'def':
        'a flag usually bearing a white skull and crossbones on a black background',
    'name':
        'pirate_flag'
}, {
    'frequency': 'c',
    'id': 108,
    'synset': 'black_sheep.n.02',
    'synonyms': ['black_sheep'],
    'def': 'sheep with a black coat',
    'name': 'black_sheep'
}, {
    'frequency': 'c',
    'id': 109,
    'synset': 'blackboard.n.01',
    'synonyms': ['blackboard', 'chalkboard'],
    'def': 'sheet of slate; for writing with chalk',
    'name': 'blackboard'
}, {
    'frequency': 'f',
    'id': 110,
    'synset': 'blanket.n.01',
    'synonyms': ['blanket'],
    'def': 'bedding that keeps a person warm in bed',
    'name': 'blanket'
}, {
    'frequency':
        'c',
    'id':
        111,
    'synset':
        'blazer.n.01',
    'synonyms': [
        'blazer', 'sport_jacket', 'sport_coat', 'sports_jacket', 'sports_coat'
    ],
    'def':
        'lightweight jacket; often striped in the colors of a club or school',
    'name':
        'blazer'
}, {
    'frequency': 'f',
    'id': 112,
    'synset': 'blender.n.01',
    'synonyms': ['blender', 'liquidizer', 'liquidiser'],
    'def': 'an electrically powered mixer that mix or chop or liquefy foods',
    'name': 'blender'
}, {
    'frequency':
        'r',
    'id':
        113,
    'synset':
        'blimp.n.02',
    'synonyms': ['blimp'],
    'def':
        'a small nonrigid airship used for observation or as a barrage balloon',
    'name':
        'blimp'
}, {
    'frequency':
        'c',
    'id':
        114,
    'synset':
        'blinker.n.01',
    'synonyms': ['blinker', 'flasher'],
    'def':
        'a light that flashes on and off; used as a signal or to send messages',
    'name':
        'blinker'
}, {
    'frequency': 'c',
    'id': 115,
    'synset': 'blueberry.n.02',
    'synonyms': ['blueberry'],
    'def': 'sweet edible dark-blue berries of blueberry plants',
    'name': 'blueberry'
}, {
    'frequency': 'r',
    'id': 116,
    'synset': 'boar.n.02',
    'synonyms': ['boar'],
    'def': 'an uncastrated male hog',
    'name': 'boar'
}, {
    'frequency':
        'r',
    'id':
        117,
    'synset':
        'board.n.09',
    'synonyms': ['gameboard'],
    'def':
        'a flat portable surface (usually rectangular) designed for board games',
    'name':
        'gameboard'
}, {
    'frequency': 'f',
    'id': 118,
    'synset': 'boat.n.01',
    'synonyms': ['boat', 'ship_(boat)'],
    'def': 'a vessel for travel on water',
    'name': 'boat'
}, {
    'frequency':
        'c',
    'id':
        119,
    'synset':
        'bobbin.n.01',
    'synonyms': ['bobbin', 'spool', 'reel'],
    'def':
        'a thing around which thread/tape/film or other flexible materials can be wound',
    'name':
        'bobbin'
}, {
    'frequency': 'r',
    'id': 120,
    'synset': 'bobby_pin.n.01',
    'synonyms': ['bobby_pin', 'hairgrip'],
    'def': 'a flat wire hairpin used to hold bobbed hair in place',
    'name': 'bobby_pin'
}, {
    'frequency': 'c',
    'id': 121,
    'synset': 'boiled_egg.n.01',
    'synonyms': ['boiled_egg', 'coddled_egg'],
    'def': 'egg cooked briefly in the shell in gently boiling water',
    'name': 'boiled_egg'
}, {
    'frequency':
        'r',
    'id':
        122,
    'synset':
        'bolo_tie.n.01',
    'synonyms': ['bolo_tie', 'bolo', 'bola_tie', 'bola'],
    'def':
        'a cord fastened around the neck with an ornamental clasp and worn as a necktie',
    'name':
        'bolo_tie'
}, {
    'frequency': 'c',
    'id': 123,
    'synset': 'bolt.n.03',
    'synonyms': ['deadbolt'],
    'def': 'the part of a lock that is engaged or withdrawn with a key',
    'name': 'deadbolt'
}, {
    'frequency': 'f',
    'id': 124,
    'synset': 'bolt.n.06',
    'synonyms': ['bolt'],
    'def': 'a screw that screws into a nut to form a fastener',
    'name': 'bolt'
}, {
    'frequency': 'r',
    'id': 125,
    'synset': 'bonnet.n.01',
    'synonyms': ['bonnet'],
    'def': 'a hat tied under the chin',
    'name': 'bonnet'
}, {
    'frequency': 'f',
    'id': 126,
    'synset': 'book.n.01',
    'synonyms': ['book'],
    'def': 'a written work or composition that has been published',
    'name': 'book'
}, {
    'frequency': 'r',
    'id': 127,
    'synset': 'book_bag.n.01',
    'synonyms': ['book_bag'],
    'def': 'a bag in which students carry their books',
    'name': 'book_bag'
}, {
    'frequency': 'c',
    'id': 128,
    'synset': 'bookcase.n.01',
    'synonyms': ['bookcase'],
    'def': 'a piece of furniture with shelves for storing books',
    'name': 'bookcase'
}, {
    'frequency': 'c',
    'id': 129,
    'synset': 'booklet.n.01',
    'synonyms': ['booklet', 'brochure', 'leaflet', 'pamphlet'],
    'def': 'a small book usually having a paper cover',
    'name': 'booklet'
}, {
    'frequency':
        'r',
    'id':
        130,
    'synset':
        'bookmark.n.01',
    'synonyms': ['bookmark', 'bookmarker'],
    'def':
        'a marker (a piece of paper or ribbon) placed between the pages of a book',
    'name':
        'bookmark'
}, {
    'frequency':
        'r',
    'id':
        131,
    'synset':
        'boom.n.04',
    'synonyms': ['boom_microphone', 'microphone_boom'],
    'def':
        'a pole carrying an overhead microphone projected over a film or tv set',
    'name':
        'boom_microphone'
}, {
    'frequency': 'f',
    'id': 132,
    'synset': 'boot.n.01',
    'synonyms': ['boot'],
    'def': 'footwear that covers the whole foot and lower leg',
    'name': 'boot'
}, {
    'frequency': 'f',
    'id': 133,
    'synset': 'bottle.n.01',
    'synonyms': ['bottle'],
    'def': 'a glass or plastic vessel used for storing drinks or other liquids',
    'name': 'bottle'
}, {
    'frequency': 'c',
    'id': 134,
    'synset': 'bottle_opener.n.01',
    'synonyms': ['bottle_opener'],
    'def': 'an opener for removing caps or corks from bottles',
    'name': 'bottle_opener'
}, {
    'frequency': 'c',
    'id': 135,
    'synset': 'bouquet.n.01',
    'synonyms': ['bouquet'],
    'def': 'an arrangement of flowers that is usually given as a present',
    'name': 'bouquet'
}, {
    'frequency': 'r',
    'id': 136,
    'synset': 'bow.n.04',
    'synonyms': ['bow_(weapon)'],
    'def': 'a weapon for shooting arrows',
    'name': 'bow_(weapon)'
}, {
    'frequency': 'f',
    'id': 137,
    'synset': 'bow.n.08',
    'synonyms': ['bow_(decorative_ribbons)'],
    'def': 'a decorative interlacing of ribbons',
    'name': 'bow_(decorative_ribbons)'
}, {
    'frequency': 'f',
    'id': 138,
    'synset': 'bow_tie.n.01',
    'synonyms': ['bow-tie', 'bowtie'],
    'def': "a man's tie that ties in a bow",
    'name': 'bow-tie'
}, {
    'frequency': 'f',
    'id': 139,
    'synset': 'bowl.n.03',
    'synonyms': ['bowl'],
    'def': 'a dish that is round and open at the top for serving foods',
    'name': 'bowl'
}, {
    'frequency':
        'r',
    'id':
        140,
    'synset':
        'bowl.n.08',
    'synonyms': ['pipe_bowl'],
    'def':
        'a small round container that is open at the top for holding tobacco',
    'name':
        'pipe_bowl'
}, {
    'frequency': 'c',
    'id': 141,
    'synset': 'bowler_hat.n.01',
    'synonyms': ['bowler_hat', 'bowler', 'derby_hat', 'derby', 'plug_hat'],
    'def': 'a felt hat that is round and hard with a narrow brim',
    'name': 'bowler_hat'
}, {
    'frequency': 'r',
    'id': 142,
    'synset': 'bowling_ball.n.01',
    'synonyms': ['bowling_ball'],
    'def': 'a large ball with finger holes used in the sport of bowling',
    'name': 'bowling_ball'
}, {
    'frequency': 'r',
    'id': 143,
    'synset': 'bowling_pin.n.01',
    'synonyms': ['bowling_pin'],
    'def': 'a club-shaped wooden object used in bowling',
    'name': 'bowling_pin'
}, {
    'frequency':
        'r',
    'id':
        144,
    'synset':
        'boxing_glove.n.01',
    'synonyms': ['boxing_glove'],
    'def':
        'large glove coverings the fists of a fighter worn for the sport of boxing',
    'name':
        'boxing_glove'
}, {
    'frequency': 'c',
    'id': 145,
    'synset': 'brace.n.06',
    'synonyms': ['suspenders'],
    'def': 'elastic straps that hold trousers up (usually used in the plural)',
    'name': 'suspenders'
}, {
    'frequency': 'f',
    'id': 146,
    'synset': 'bracelet.n.02',
    'synonyms': ['bracelet', 'bangle'],
    'def': 'jewelry worn around the wrist for decoration',
    'name': 'bracelet'
}, {
    'frequency': 'r',
    'id': 147,
    'synset': 'brass.n.07',
    'synonyms': ['brass_plaque'],
    'def': 'a memorial made of brass',
    'name': 'brass_plaque'
}, {
    'frequency': 'c',
    'id': 148,
    'synset': 'brassiere.n.01',
    'synonyms': ['brassiere', 'bra', 'bandeau'],
    'def': 'an undergarment worn by women to support their breasts',
    'name': 'brassiere'
}, {
    'frequency': 'c',
    'id': 149,
    'synset': 'bread-bin.n.01',
    'synonyms': ['bread-bin', 'breadbox'],
    'def': 'a container used to keep bread or cake in',
    'name': 'bread-bin'
}, {
    'frequency': 'r',
    'id': 150,
    'synset': 'breechcloth.n.01',
    'synonyms': ['breechcloth', 'breechclout', 'loincloth'],
    'def': 'a garment that provides covering for the loins',
    'name': 'breechcloth'
}, {
    'frequency': 'c',
    'id': 151,
    'synset': 'bridal_gown.n.01',
    'synonyms': ['bridal_gown', 'wedding_gown', 'wedding_dress'],
    'def': 'a gown worn by the bride at a wedding',
    'name': 'bridal_gown'
}, {
    'frequency': 'c',
    'id': 152,
    'synset': 'briefcase.n.01',
    'synonyms': ['briefcase'],
    'def': 'a case with a handle; for carrying papers or files or books',
    'name': 'briefcase'
}, {
    'frequency':
        'c',
    'id':
        153,
    'synset':
        'bristle_brush.n.01',
    'synonyms': ['bristle_brush'],
    'def':
        'a brush that is made with the short stiff hairs of an animal or plant',
    'name':
        'bristle_brush'
}, {
    'frequency': 'f',
    'id': 154,
    'synset': 'broccoli.n.01',
    'synonyms': ['broccoli'],
    'def': 'plant with dense clusters of tight green flower buds',
    'name': 'broccoli'
}, {
    'frequency': 'r',
    'id': 155,
    'synset': 'brooch.n.01',
    'synonyms': ['broach'],
    'def': 'a decorative pin worn by women',
    'name': 'broach'
}, {
    'frequency':
        'c',
    'id':
        156,
    'synset':
        'broom.n.01',
    'synonyms': ['broom'],
    'def':
        'bundle of straws or twigs attached to a long handle; used for cleaning',
    'name':
        'broom'
}, {
    'frequency': 'c',
    'id': 157,
    'synset': 'brownie.n.03',
    'synonyms': ['brownie'],
    'def': 'square or bar of very rich chocolate cake usually with nuts',
    'name': 'brownie'
}, {
    'frequency': 'c',
    'id': 158,
    'synset': 'brussels_sprouts.n.01',
    'synonyms': ['brussels_sprouts'],
    'def': 'the small edible cabbage-like buds growing along a stalk',
    'name': 'brussels_sprouts'
}, {
    'frequency': 'r',
    'id': 159,
    'synset': 'bubble_gum.n.01',
    'synonyms': ['bubble_gum'],
    'def': 'a kind of chewing gum that can be blown into bubbles',
    'name': 'bubble_gum'
}, {
    'frequency': 'f',
    'id': 160,
    'synset': 'bucket.n.01',
    'synonyms': ['bucket', 'pail'],
    'def': 'a roughly cylindrical vessel that is open at the top',
    'name': 'bucket'
}, {
    'frequency': 'r',
    'id': 161,
    'synset': 'buggy.n.01',
    'synonyms': ['horse_buggy'],
    'def': 'a small lightweight carriage; drawn by a single horse',
    'name': 'horse_buggy'
}, {
    'frequency': 'c',
    'id': 162,
    'synset': 'bull.n.11',
    'synonyms': ['bull'],
    'def': 'mature male cow',
    'name': 'bull'
}, {
    'frequency':
        'r',
    'id':
        163,
    'synset':
        'bulldog.n.01',
    'synonyms': ['bulldog'],
    'def':
        'a thickset short-haired dog with a large head and strong undershot lower jaw',
    'name':
        'bulldog'
}, {
    'frequency':
        'r',
    'id':
        164,
    'synset':
        'bulldozer.n.01',
    'synonyms': ['bulldozer', 'dozer'],
    'def':
        'large powerful tractor; a large blade in front flattens areas of ground',
    'name':
        'bulldozer'
}, {
    'frequency': 'c',
    'id': 165,
    'synset': 'bullet_train.n.01',
    'synonyms': ['bullet_train'],
    'def': 'a high-speed passenger train',
    'name': 'bullet_train'
}, {
    'frequency': 'c',
    'id': 166,
    'synset': 'bulletin_board.n.02',
    'synonyms': ['bulletin_board', 'notice_board'],
    'def': 'a board that hangs on a wall; displays announcements',
    'name': 'bulletin_board'
}, {
    'frequency': 'r',
    'id': 167,
    'synset': 'bulletproof_vest.n.01',
    'synonyms': ['bulletproof_vest'],
    'def': 'a vest capable of resisting the impact of a bullet',
    'name': 'bulletproof_vest'
}, {
    'frequency': 'c',
    'id': 168,
    'synset': 'bullhorn.n.01',
    'synonyms': ['bullhorn', 'megaphone'],
    'def': 'a portable loudspeaker with built-in microphone and amplifier',
    'name': 'bullhorn'
}, {
    'frequency': 'r',
    'id': 169,
    'synset': 'bully_beef.n.01',
    'synonyms': ['corned_beef', 'corn_beef'],
    'def': 'beef cured or pickled in brine',
    'name': 'corned_beef'
}, {
    'frequency': 'f',
    'id': 170,
    'synset': 'bun.n.01',
    'synonyms': ['bun', 'roll'],
    'def': 'small rounded bread either plain or sweet',
    'name': 'bun'
}, {
    'frequency': 'c',
    'id': 171,
    'synset': 'bunk_bed.n.01',
    'synonyms': ['bunk_bed'],
    'def': 'beds built one above the other',
    'name': 'bunk_bed'
}, {
    'frequency':
        'f',
    'id':
        172,
    'synset':
        'buoy.n.01',
    'synonyms': ['buoy'],
    'def':
        'a float attached by rope to the seabed to mark channels in a harbor or underwater hazards',
    'name':
        'buoy'
}, {
    'frequency': 'r',
    'id': 173,
    'synset': 'burrito.n.01',
    'synonyms': ['burrito'],
    'def': 'a flour tortilla folded around a filling',
    'name': 'burrito'
}, {
    'frequency': 'f',
    'id': 174,
    'synset': 'bus.n.01',
    'synonyms': [
        'bus_(vehicle)', 'autobus', 'charabanc', 'double-decker', 'motorbus',
        'motorcoach'
    ],
    'def': 'a vehicle carrying many passengers; used for public transport',
    'name': 'bus_(vehicle)'
}, {
    'frequency':
        'c',
    'id':
        175,
    'synset':
        'business_card.n.01',
    'synonyms': ['business_card'],
    'def':
        "a card on which are printed the person's name and business affiliation",
    'name':
        'business_card'
}, {
    'frequency': 'c',
    'id': 176,
    'synset': 'butcher_knife.n.01',
    'synonyms': ['butcher_knife'],
    'def': 'a large sharp knife for cutting or trimming meat',
    'name': 'butcher_knife'
}, {
    'frequency':
        'c',
    'id':
        177,
    'synset':
        'butter.n.01',
    'synonyms': ['butter'],
    'def':
        'an edible emulsion of fat globules made by churning milk or cream; for cooking and table use',
    'name':
        'butter'
}, {
    'frequency':
        'c',
    'id':
        178,
    'synset':
        'butterfly.n.01',
    'synonyms': ['butterfly'],
    'def':
        'insect typically having a slender body with knobbed antennae and broad colorful wings',
    'name':
        'butterfly'
}, {
    'frequency':
        'f',
    'id':
        179,
    'synset':
        'button.n.01',
    'synonyms': ['button'],
    'def':
        'a round fastener sewn to shirts and coats etc to fit through buttonholes',
    'name':
        'button'
}, {
    'frequency':
        'f',
    'id':
        180,
    'synset':
        'cab.n.03',
    'synonyms': ['cab_(taxi)', 'taxi', 'taxicab'],
    'def':
        'a car that takes passengers where they want to go in exchange for money',
    'name':
        'cab_(taxi)'
}, {
    'frequency':
        'r',
    'id':
        181,
    'synset':
        'cabana.n.01',
    'synonyms': ['cabana'],
    'def':
        'a small tent used as a dressing room beside the sea or a swimming pool',
    'name':
        'cabana'
}, {
    'frequency':
        'r',
    'id':
        182,
    'synset':
        'cabin_car.n.01',
    'synonyms': ['cabin_car', 'caboose'],
    'def':
        'a car on a freight train for use of the train crew; usually the last car on the train',
    'name':
        'cabin_car'
}, {
    'frequency':
        'f',
    'id':
        183,
    'synset':
        'cabinet.n.01',
    'synonyms': ['cabinet'],
    'def':
        'a piece of furniture resembling a cupboard with doors and shelves and drawers',
    'name':
        'cabinet'
}, {
    'frequency':
        'r',
    'id':
        184,
    'synset':
        'cabinet.n.03',
    'synonyms': ['locker', 'storage_locker'],
    'def':
        'a storage compartment for clothes and valuables; usually it has a lock',
    'name':
        'locker'
}, {
    'frequency':
        'f',
    'id':
        185,
    'synset':
        'cake.n.03',
    'synonyms': ['cake'],
    'def':
        'baked goods made from or based on a mixture of flour, sugar, eggs, and fat',
    'name':
        'cake'
}, {
    'frequency': 'c',
    'id': 186,
    'synset': 'calculator.n.02',
    'synonyms': ['calculator'],
    'def': 'a small machine that is used for mathematical calculations',
    'name': 'calculator'
}, {
    'frequency':
        'f',
    'id':
        187,
    'synset':
        'calendar.n.02',
    'synonyms': ['calendar'],
    'def':
        'a list or register of events (appointments/social events/court cases, etc)',
    'name':
        'calendar'
}, {
    'frequency': 'c',
    'id': 188,
    'synset': 'calf.n.01',
    'synonyms': ['calf'],
    'def': 'young of domestic cattle',
    'name': 'calf'
}, {
    'frequency': 'c',
    'id': 189,
    'synset': 'camcorder.n.01',
    'synonyms': ['camcorder'],
    'def': 'a portable television camera and videocassette recorder',
    'name': 'camcorder'
}, {
    'frequency':
        'c',
    'id':
        190,
    'synset':
        'camel.n.01',
    'synonyms': ['camel'],
    'def':
        'cud-chewing mammal used as a draft or saddle animal in desert regions',
    'name':
        'camel'
}, {
    'frequency': 'f',
    'id': 191,
    'synset': 'camera.n.01',
    'synonyms': ['camera'],
    'def': 'equipment for taking photographs',
    'name': 'camera'
}, {
    'frequency': 'c',
    'id': 192,
    'synset': 'camera_lens.n.01',
    'synonyms': ['camera_lens'],
    'def': 'a lens that focuses the image in a camera',
    'name': 'camera_lens'
}, {
    'frequency': 'c',
    'id': 193,
    'synset': 'camper.n.02',
    'synonyms': ['camper_(vehicle)', 'camping_bus', 'motor_home'],
    'def': 'a recreational vehicle equipped for camping out while traveling',
    'name': 'camper_(vehicle)'
}, {
    'frequency': 'f',
    'id': 194,
    'synset': 'can.n.01',
    'synonyms': ['can', 'tin_can'],
    'def': 'airtight sealed metal container for food or drink or paint etc.',
    'name': 'can'
}, {
    'frequency': 'c',
    'id': 195,
    'synset': 'can_opener.n.01',
    'synonyms': ['can_opener', 'tin_opener'],
    'def': 'a device for cutting cans open',
    'name': 'can_opener'
}, {
    'frequency': 'r',
    'id': 196,
    'synset': 'candelabrum.n.01',
    'synonyms': ['candelabrum', 'candelabra'],
    'def': 'branched candlestick; ornamental; has several lights',
    'name': 'candelabrum'
}, {
    'frequency': 'f',
    'id': 197,
    'synset': 'candle.n.01',
    'synonyms': ['candle', 'candlestick'],
    'def': 'stick of wax with a wick in the middle',
    'name': 'candle'
}, {
    'frequency': 'f',
    'id': 198,
    'synset': 'candlestick.n.01',
    'synonyms': ['candle_holder'],
    'def': 'a holder with sockets for candles',
    'name': 'candle_holder'
}, {
    'frequency': 'r',
    'id': 199,
    'synset': 'candy_bar.n.01',
    'synonyms': ['candy_bar'],
    'def': 'a candy shaped as a bar',
    'name': 'candy_bar'
}, {
    'frequency': 'c',
    'id': 200,
    'synset': 'candy_cane.n.01',
    'synonyms': ['candy_cane'],
    'def': 'a hard candy in the shape of a rod (usually with stripes)',
    'name': 'candy_cane'
}, {
    'frequency': 'c',
    'id': 201,
    'synset': 'cane.n.01',
    'synonyms': ['walking_cane'],
    'def': 'a stick that people can lean on to help them walk',
    'name': 'walking_cane'
}, {
    'frequency': 'c',
    'id': 202,
    'synset': 'canister.n.02',
    'synonyms': ['canister', 'cannister'],
    'def': 'metal container for storing dry foods such as tea or flour',
    'name': 'canister'
}, {
    'frequency': 'r',
    'id': 203,
    'synset': 'cannon.n.02',
    'synonyms': ['cannon'],
    'def': 'heavy gun fired from a tank',
    'name': 'cannon'
}, {
    'frequency':
        'c',
    'id':
        204,
    'synset':
        'canoe.n.01',
    'synonyms': ['canoe'],
    'def':
        'small and light boat; pointed at both ends; propelled with a paddle',
    'name':
        'canoe'
}, {
    'frequency':
        'r',
    'id':
        205,
    'synset':
        'cantaloup.n.02',
    'synonyms': ['cantaloup', 'cantaloupe'],
    'def':
        'the fruit of a cantaloup vine; small to medium-sized melon with yellowish flesh',
    'name':
        'cantaloup'
}, {
    'frequency': 'r',
    'id': 206,
    'synset': 'canteen.n.01',
    'synonyms': ['canteen'],
    'def': 'a flask for carrying water; used by soldiers or travelers',
    'name': 'canteen'
}, {
    'frequency': 'c',
    'id': 207,
    'synset': 'cap.n.01',
    'synonyms': ['cap_(headwear)'],
    'def': 'a tight-fitting headwear',
    'name': 'cap_(headwear)'
}, {
    'frequency': 'f',
    'id': 208,
    'synset': 'cap.n.02',
    'synonyms': ['bottle_cap', 'cap_(container_lid)'],
    'def': 'a top (as for a bottle)',
    'name': 'bottle_cap'
}, {
    'frequency': 'r',
    'id': 209,
    'synset': 'cape.n.02',
    'synonyms': ['cape'],
    'def': 'a sleeveless garment like a cloak but shorter',
    'name': 'cape'
}, {
    'frequency': 'c',
    'id': 210,
    'synset': 'cappuccino.n.01',
    'synonyms': ['cappuccino', 'coffee_cappuccino'],
    'def': 'equal parts of espresso and steamed milk',
    'name': 'cappuccino'
}, {
    'frequency': 'f',
    'id': 211,
    'synset': 'car.n.01',
    'synonyms': ['car_(automobile)', 'auto_(automobile)', 'automobile'],
    'def': 'a motor vehicle with four wheels',
    'name': 'car_(automobile)'
}, {
    'frequency': 'f',
    'id': 212,
    'synset': 'car.n.02',
    'synonyms': [
        'railcar_(part_of_a_train)', 'railway_car_(part_of_a_train)',
        'railroad_car_(part_of_a_train)'
    ],
    'def': 'a wheeled vehicle adapted to the rails of railroad',
    'name': 'railcar_(part_of_a_train)'
}, {
    'frequency': 'r',
    'id': 213,
    'synset': 'car.n.04',
    'synonyms': ['elevator_car'],
    'def': 'where passengers ride up and down',
    'name': 'elevator_car'
}, {
    'frequency': 'r',
    'id': 214,
    'synset': 'car_battery.n.01',
    'synonyms': ['car_battery', 'automobile_battery'],
    'def': 'a battery in a motor vehicle',
    'name': 'car_battery'
}, {
    'frequency': 'c',
    'id': 215,
    'synset': 'card.n.02',
    'synonyms': ['identity_card'],
    'def': 'a card certifying the identity of the bearer',
    'name': 'identity_card'
}, {
    'frequency':
        'c',
    'id':
        216,
    'synset':
        'card.n.03',
    'synonyms': ['card'],
    'def':
        'a rectangular piece of paper used to send messages (e.g. greetings or pictures)',
    'name':
        'card'
}, {
    'frequency':
        'r',
    'id':
        217,
    'synset':
        'cardigan.n.01',
    'synonyms': ['cardigan'],
    'def':
        'knitted jacket that is fastened up the front with buttons or a zipper',
    'name':
        'cardigan'
}, {
    'frequency': 'r',
    'id': 218,
    'synset': 'cargo_ship.n.01',
    'synonyms': ['cargo_ship', 'cargo_vessel'],
    'def': 'a ship designed to carry cargo',
    'name': 'cargo_ship'
}, {
    'frequency': 'r',
    'id': 219,
    'synset': 'carnation.n.01',
    'synonyms': ['carnation'],
    'def': 'plant with pink to purple-red spice-scented usually double flowers',
    'name': 'carnation'
}, {
    'frequency': 'c',
    'id': 220,
    'synset': 'carriage.n.02',
    'synonyms': ['horse_carriage'],
    'def': 'a vehicle with wheels drawn by one or more horses',
    'name': 'horse_carriage'
}, {
    'frequency': 'f',
    'id': 221,
    'synset': 'carrot.n.01',
    'synonyms': ['carrot'],
    'def': 'deep orange edible root of the cultivated carrot plant',
    'name': 'carrot'
}, {
    'frequency': 'c',
    'id': 222,
    'synset': 'carryall.n.01',
    'synonyms': ['tote_bag'],
    'def': 'a capacious bag or basket',
    'name': 'tote_bag'
}, {
    'frequency':
        'c',
    'id':
        223,
    'synset':
        'cart.n.01',
    'synonyms': ['cart'],
    'def':
        'a heavy open wagon usually having two wheels and drawn by an animal',
    'name':
        'cart'
}, {
    'frequency': 'c',
    'id': 224,
    'synset': 'carton.n.02',
    'synonyms': ['carton'],
    'def': 'a box made of cardboard; opens by flaps on top',
    'name': 'carton'
}, {
    'frequency': 'c',
    'id': 225,
    'synset': 'cash_register.n.01',
    'synonyms': ['cash_register', 'register_(for_cash_transactions)'],
    'def': 'a cashbox with an adding machine to register transactions',
    'name': 'cash_register'
}, {
    'frequency': 'r',
    'id': 226,
    'synset': 'casserole.n.01',
    'synonyms': ['casserole'],
    'def': 'food cooked and served in a casserole',
    'name': 'casserole'
}, {
    'frequency':
        'r',
    'id':
        227,
    'synset':
        'cassette.n.01',
    'synonyms': ['cassette'],
    'def':
        'a container that holds a magnetic tape used for recording or playing sound or video',
    'name':
        'cassette'
}, {
    'frequency':
        'c',
    'id':
        228,
    'synset':
        'cast.n.05',
    'synonyms': ['cast', 'plaster_cast', 'plaster_bandage'],
    'def':
        'bandage consisting of a firm covering that immobilizes broken bones while they heal',
    'name':
        'cast'
}, {
    'frequency': 'f',
    'id': 229,
    'synset': 'cat.n.01',
    'synonyms': ['cat'],
    'def': 'a domestic house cat',
    'name': 'cat'
}, {
    'frequency': 'c',
    'id': 230,
    'synset': 'cauliflower.n.02',
    'synonyms': ['cauliflower'],
    'def': 'edible compact head of white undeveloped flowers',
    'name': 'cauliflower'
}, {
    'frequency':
        'r',
    'id':
        231,
    'synset':
        'caviar.n.01',
    'synonyms': ['caviar', 'caviare'],
    'def':
        "salted roe of sturgeon or other large fish; usually served as an hors d'oeuvre",
    'name':
        'caviar'
}, {
    'frequency': 'c',
    'id': 232,
    'synset': 'cayenne.n.02',
    'synonyms': [
        'cayenne_(spice)', 'cayenne_pepper_(spice)', 'red_pepper_(spice)'
    ],
    'def': 'ground pods and seeds of pungent red peppers of the genus Capsicum',
    'name': 'cayenne_(spice)'
}, {
    'frequency': 'c',
    'id': 233,
    'synset': 'cd_player.n.01',
    'synonyms': ['CD_player'],
    'def': 'electronic equipment for playing compact discs (CDs)',
    'name': 'CD_player'
}, {
    'frequency':
        'c',
    'id':
        234,
    'synset':
        'celery.n.01',
    'synonyms': ['celery'],
    'def':
        'widely cultivated herb with aromatic leaf stalks that are eaten raw or cooked',
    'name':
        'celery'
}, {
    'frequency': 'f',
    'id': 235,
    'synset': 'cellular_telephone.n.01',
    'synonyms': [
        'cellular_telephone', 'cellular_phone', 'cellphone', 'mobile_phone',
        'smart_phone'
    ],
    'def': 'a hand-held mobile telephone',
    'name': 'cellular_telephone'
}, {
    'frequency': 'r',
    'id': 236,
    'synset': 'chain_mail.n.01',
    'synonyms': [
        'chain_mail', 'ring_mail', 'chain_armor', 'chain_armour', 'ring_armor',
        'ring_armour'
    ],
    'def': '(Middle Ages) flexible armor made of interlinked metal rings',
    'name': 'chain_mail'
}, {
    'frequency': 'f',
    'id': 237,
    'synset': 'chair.n.01',
    'synonyms': ['chair'],
    'def': 'a seat for one person, with a support for the back',
    'name': 'chair'
}, {
    'frequency': 'r',
    'id': 238,
    'synset': 'chaise_longue.n.01',
    'synonyms': ['chaise_longue', 'chaise', 'daybed'],
    'def': 'a long chair; for reclining',
    'name': 'chaise_longue'
}, {
    'frequency':
        'r',
    'id':
        239,
    'synset':
        'champagne.n.01',
    'synonyms': ['champagne'],
    'def':
        'a white sparkling wine produced in Champagne or resembling that produced there',
    'name':
        'champagne'
}, {
    'frequency': 'f',
    'id': 240,
    'synset': 'chandelier.n.01',
    'synonyms': ['chandelier'],
    'def': 'branched lighting fixture; often ornate; hangs from the ceiling',
    'name': 'chandelier'
}, {
    'frequency':
        'r',
    'id':
        241,
    'synset':
        'chap.n.04',
    'synonyms': ['chap'],
    'def':
        'leather leggings without a seat; worn over trousers by cowboys to protect their legs',
    'name':
        'chap'
}, {
    'frequency': 'r',
    'id': 242,
    'synset': 'checkbook.n.01',
    'synonyms': ['checkbook', 'chequebook'],
    'def': 'a book issued to holders of checking accounts',
    'name': 'checkbook'
}, {
    'frequency': 'r',
    'id': 243,
    'synset': 'checkerboard.n.01',
    'synonyms': ['checkerboard'],
    'def': 'a board having 64 squares of two alternating colors',
    'name': 'checkerboard'
}, {
    'frequency': 'c',
    'id': 244,
    'synset': 'cherry.n.03',
    'synonyms': ['cherry'],
    'def': 'a red fruit with a single hard stone',
    'name': 'cherry'
}, {
    'frequency': 'r',
    'id': 245,
    'synset': 'chessboard.n.01',
    'synonyms': ['chessboard'],
    'def': 'a checkerboard used to play chess',
    'name': 'chessboard'
}, {
    'frequency': 'r',
    'id': 246,
    'synset': 'chest_of_drawers.n.01',
    'synonyms': [
        'chest_of_drawers_(furniture)', 'bureau_(furniture)',
        'chest_(furniture)'
    ],
    'def': 'furniture with drawers for keeping clothes',
    'name': 'chest_of_drawers_(furniture)'
}, {
    'frequency': 'c',
    'id': 247,
    'synset': 'chicken.n.02',
    'synonyms': ['chicken_(animal)'],
    'def': 'a domestic fowl bred for flesh or eggs',
    'name': 'chicken_(animal)'
}, {
    'frequency':
        'c',
    'id':
        248,
    'synset':
        'chicken_wire.n.01',
    'synonyms': ['chicken_wire'],
    'def':
        'a galvanized wire network with a hexagonal mesh; used to build fences',
    'name':
        'chicken_wire'
}, {
    'frequency': 'r',
    'id': 249,
    'synset': 'chickpea.n.01',
    'synonyms': ['chickpea', 'garbanzo'],
    'def': 'the seed of the chickpea plant; usually dried',
    'name': 'chickpea'
}, {
    'frequency':
        'r',
    'id':
        250,
    'synset':
        'chihuahua.n.03',
    'synonyms': ['Chihuahua'],
    'def':
        'an old breed of tiny short-haired dog with protruding eyes from Mexico',
    'name':
        'Chihuahua'
}, {
    'frequency': 'r',
    'id': 251,
    'synset': 'chili.n.02',
    'synonyms': [
        'chili_(vegetable)', 'chili_pepper_(vegetable)', 'chilli_(vegetable)',
        'chilly_(vegetable)', 'chile_(vegetable)'
    ],
    'def': 'very hot and finely tapering pepper of special pungency',
    'name': 'chili_(vegetable)'
}, {
    'frequency':
        'r',
    'id':
        252,
    'synset':
        'chime.n.01',
    'synonyms': ['chime', 'gong'],
    'def':
        'an instrument consisting of a set of bells that are struck with a hammer',
    'name':
        'chime'
}, {
    'frequency': 'r',
    'id': 253,
    'synset': 'chinaware.n.01',
    'synonyms': ['chinaware'],
    'def': 'dishware made of high quality porcelain',
    'name': 'chinaware'
}, {
    'frequency': 'c',
    'id': 254,
    'synset': 'chip.n.04',
    'synonyms': ['crisp_(potato_chip)', 'potato_chip'],
    'def': 'a thin crisp slice of potato fried in deep fat',
    'name': 'crisp_(potato_chip)'
}, {
    'frequency': 'r',
    'id': 255,
    'synset': 'chip.n.06',
    'synonyms': ['poker_chip'],
    'def': 'a small disk-shaped counter used to represent money when gambling',
    'name': 'poker_chip'
}, {
    'frequency': 'c',
    'id': 256,
    'synset': 'chocolate_bar.n.01',
    'synonyms': ['chocolate_bar'],
    'def': 'a bar of chocolate candy',
    'name': 'chocolate_bar'
}, {
    'frequency': 'c',
    'id': 257,
    'synset': 'chocolate_cake.n.01',
    'synonyms': ['chocolate_cake'],
    'def': 'cake containing chocolate',
    'name': 'chocolate_cake'
}, {
    'frequency': 'r',
    'id': 258,
    'synset': 'chocolate_milk.n.01',
    'synonyms': ['chocolate_milk'],
    'def': 'milk flavored with chocolate syrup',
    'name': 'chocolate_milk'
}, {
    'frequency': 'r',
    'id': 259,
    'synset': 'chocolate_mousse.n.01',
    'synonyms': ['chocolate_mousse'],
    'def': 'dessert mousse made with chocolate',
    'name': 'chocolate_mousse'
}, {
    'frequency': 'f',
    'id': 260,
    'synset': 'choker.n.03',
    'synonyms': ['choker', 'collar', 'neckband'],
    'def': 'necklace that fits tightly around the neck',
    'name': 'choker'
}, {
    'frequency': 'f',
    'id': 261,
    'synset': 'chopping_board.n.01',
    'synonyms': ['chopping_board', 'cutting_board', 'chopping_block'],
    'def': 'a wooden board where meats or vegetables can be cut',
    'name': 'chopping_board'
}, {
    'frequency':
        'c',
    'id':
        262,
    'synset':
        'chopstick.n.01',
    'synonyms': ['chopstick'],
    'def':
        'one of a pair of slender sticks used as oriental tableware to eat food with',
    'name':
        'chopstick'
}, {
    'frequency': 'f',
    'id': 263,
    'synset': 'christmas_tree.n.05',
    'synonyms': ['Christmas_tree'],
    'def': 'an ornamented evergreen used as a Christmas decoration',
    'name': 'Christmas_tree'
}, {
    'frequency': 'c',
    'id': 264,
    'synset': 'chute.n.02',
    'synonyms': ['slide'],
    'def': 'sloping channel through which things can descend',
    'name': 'slide'
}, {
    'frequency': 'r',
    'id': 265,
    'synset': 'cider.n.01',
    'synonyms': ['cider', 'cyder'],
    'def': 'a beverage made from juice pressed from apples',
    'name': 'cider'
}, {
    'frequency': 'r',
    'id': 266,
    'synset': 'cigar_box.n.01',
    'synonyms': ['cigar_box'],
    'def': 'a box for holding cigars',
    'name': 'cigar_box'
}, {
    'frequency': 'c',
    'id': 267,
    'synset': 'cigarette.n.01',
    'synonyms': ['cigarette'],
    'def': 'finely ground tobacco wrapped in paper; for smoking',
    'name': 'cigarette'
}, {
    'frequency': 'c',
    'id': 268,
    'synset': 'cigarette_case.n.01',
    'synonyms': ['cigarette_case', 'cigarette_pack'],
    'def': 'a small flat case for holding cigarettes',
    'name': 'cigarette_case'
}, {
    'frequency': 'f',
    'id': 269,
    'synset': 'cistern.n.02',
    'synonyms': ['cistern', 'water_tank'],
    'def': 'a tank that holds the water used to flush a toilet',
    'name': 'cistern'
}, {
    'frequency': 'r',
    'id': 270,
    'synset': 'clarinet.n.01',
    'synonyms': ['clarinet'],
    'def': 'a single-reed instrument with a straight tube',
    'name': 'clarinet'
}, {
    'frequency':
        'r',
    'id':
        271,
    'synset':
        'clasp.n.01',
    'synonyms': ['clasp'],
    'def':
        'a fastener (as a buckle or hook) that is used to hold two things together',
    'name':
        'clasp'
}, {
    'frequency': 'c',
    'id': 272,
    'synset': 'cleansing_agent.n.01',
    'synonyms': ['cleansing_agent', 'cleanser', 'cleaner'],
    'def': 'a preparation used in cleaning something',
    'name': 'cleansing_agent'
}, {
    'frequency': 'r',
    'id': 273,
    'synset': 'clementine.n.01',
    'synonyms': ['clementine'],
    'def': 'a variety of mandarin orange',
    'name': 'clementine'
}, {
    'frequency':
        'c',
    'id':
        274,
    'synset':
        'clip.n.03',
    'synonyms': ['clip'],
    'def':
        'any of various small fasteners used to hold loose articles together',
    'name':
        'clip'
}, {
    'frequency': 'c',
    'id': 275,
    'synset': 'clipboard.n.01',
    'synonyms': ['clipboard'],
    'def': 'a small writing board with a clip at the top for holding papers',
    'name': 'clipboard'
}, {
    'frequency': 'f',
    'id': 276,
    'synset': 'clock.n.01',
    'synonyms': ['clock', 'timepiece', 'timekeeper'],
    'def': 'a timepiece that shows the time of day',
    'name': 'clock'
}, {
    'frequency': 'f',
    'id': 277,
    'synset': 'clock_tower.n.01',
    'synonyms': ['clock_tower'],
    'def': 'a tower with a large clock visible high up on an outside face',
    'name': 'clock_tower'
}, {
    'frequency':
        'c',
    'id':
        278,
    'synset':
        'clothes_hamper.n.01',
    'synonyms': ['clothes_hamper', 'laundry_basket', 'clothes_basket'],
    'def':
        'a hamper that holds dirty clothes to be washed or wet clothes to be dried',
    'name':
        'clothes_hamper'
}, {
    'frequency': 'c',
    'id': 279,
    'synset': 'clothespin.n.01',
    'synonyms': ['clothespin', 'clothes_peg'],
    'def': 'wood or plastic fastener; for holding clothes on a clothesline',
    'name': 'clothespin'
}, {
    'frequency': 'r',
    'id': 280,
    'synset': 'clutch_bag.n.01',
    'synonyms': ['clutch_bag'],
    'def': "a woman's strapless purse that is carried in the hand",
    'name': 'clutch_bag'
}, {
    'frequency': 'f',
    'id': 281,
    'synset': 'coaster.n.03',
    'synonyms': ['coaster'],
    'def': 'a covering (plate or mat) that protects the surface of a table',
    'name': 'coaster'
}, {
    'frequency':
        'f',
    'id':
        282,
    'synset':
        'coat.n.01',
    'synonyms': ['coat'],
    'def':
        'an outer garment that has sleeves and covers the body from shoulder down',
    'name':
        'coat'
}, {
    'frequency': 'c',
    'id': 283,
    'synset': 'coat_hanger.n.01',
    'synonyms': ['coat_hanger', 'clothes_hanger', 'dress_hanger'],
    'def': "a hanger that is shaped like a person's shoulders",
    'name': 'coat_hanger'
}, {
    'frequency': 'r',
    'id': 284,
    'synset': 'coatrack.n.01',
    'synonyms': ['coatrack', 'hatrack'],
    'def': 'a rack with hooks for temporarily holding coats and hats',
    'name': 'coatrack'
}, {
    'frequency': 'c',
    'id': 285,
    'synset': 'cock.n.04',
    'synonyms': ['cock', 'rooster'],
    'def': 'adult male chicken',
    'name': 'cock'
}, {
    'frequency': 'c',
    'id': 286,
    'synset': 'coconut.n.02',
    'synonyms': ['coconut', 'cocoanut'],
    'def': 'large hard-shelled brown oval nut with a fibrous husk',
    'name': 'coconut'
}, {
    'frequency':
        'r',
    'id':
        287,
    'synset':
        'coffee_filter.n.01',
    'synonyms': ['coffee_filter'],
    'def':
        'filter (usually of paper) that passes the coffee and retains the coffee grounds',
    'name':
        'coffee_filter'
}, {
    'frequency': 'f',
    'id': 288,
    'synset': 'coffee_maker.n.01',
    'synonyms': ['coffee_maker', 'coffee_machine'],
    'def': 'a kitchen appliance for brewing coffee automatically',
    'name': 'coffee_maker'
}, {
    'frequency':
        'f',
    'id':
        289,
    'synset':
        'coffee_table.n.01',
    'synonyms': ['coffee_table', 'cocktail_table'],
    'def':
        'low table where magazines can be placed and coffee or cocktails are served',
    'name':
        'coffee_table'
}, {
    'frequency': 'c',
    'id': 290,
    'synset': 'coffeepot.n.01',
    'synonyms': ['coffeepot'],
    'def': 'tall pot in which coffee is brewed',
    'name': 'coffeepot'
}, {
    'frequency': 'r',
    'id': 291,
    'synset': 'coil.n.05',
    'synonyms': ['coil'],
    'def': 'tubing that is wound in a spiral',
    'name': 'coil'
}, {
    'frequency': 'c',
    'id': 292,
    'synset': 'coin.n.01',
    'synonyms': ['coin'],
    'def': 'a flat metal piece (usually a disc) used as money',
    'name': 'coin'
}, {
    'frequency': 'r',
    'id': 293,
    'synset': 'colander.n.01',
    'synonyms': ['colander', 'cullender'],
    'def': 'bowl-shaped strainer; used to wash or drain foods',
    'name': 'colander'
}, {
    'frequency': 'c',
    'id': 294,
    'synset': 'coleslaw.n.01',
    'synonyms': ['coleslaw', 'slaw'],
    'def': 'basically shredded cabbage',
    'name': 'coleslaw'
}, {
    'frequency': 'r',
    'id': 295,
    'synset': 'coloring_material.n.01',
    'synonyms': ['coloring_material', 'colouring_material'],
    'def': 'any material used for its color',
    'name': 'coloring_material'
}, {
    'frequency':
        'r',
    'id':
        296,
    'synset':
        'combination_lock.n.01',
    'synonyms': ['combination_lock'],
    'def':
        'lock that can be opened only by turning dials in a special sequence',
    'name':
        'combination_lock'
}, {
    'frequency': 'c',
    'id': 297,
    'synset': 'comforter.n.04',
    'synonyms': ['pacifier', 'teething_ring'],
    'def': 'device used for an infant to suck or bite on',
    'name': 'pacifier'
}, {
    'frequency': 'r',
    'id': 298,
    'synset': 'comic_book.n.01',
    'synonyms': ['comic_book'],
    'def': 'a magazine devoted to comic strips',
    'name': 'comic_book'
}, {
    'frequency': 'f',
    'id': 299,
    'synset': 'computer_keyboard.n.01',
    'synonyms': ['computer_keyboard', 'keyboard_(computer)'],
    'def': 'a keyboard that is a data input device for computers',
    'name': 'computer_keyboard'
}, {
    'frequency':
        'r',
    'id':
        300,
    'synset':
        'concrete_mixer.n.01',
    'synonyms': ['concrete_mixer', 'cement_mixer'],
    'def':
        'a machine with a large revolving drum in which cement/concrete is mixed',
    'name':
        'concrete_mixer'
}, {
    'frequency': 'f',
    'id': 301,
    'synset': 'cone.n.01',
    'synonyms': ['cone', 'traffic_cone'],
    'def': 'a cone-shaped object used to direct traffic',
    'name': 'cone'
}, {
    'frequency': 'f',
    'id': 302,
    'synset': 'control.n.09',
    'synonyms': ['control', 'controller'],
    'def': 'a mechanism that controls the operation of a machine',
    'name': 'control'
}, {
    'frequency': 'r',
    'id': 303,
    'synset': 'convertible.n.01',
    'synonyms': ['convertible_(automobile)'],
    'def': 'a car that has top that can be folded or removed',
    'name': 'convertible_(automobile)'
}, {
    'frequency': 'r',
    'id': 304,
    'synset': 'convertible.n.03',
    'synonyms': ['sofa_bed'],
    'def': 'a sofa that can be converted into a bed',
    'name': 'sofa_bed'
}, {
    'frequency':
        'c',
    'id':
        305,
    'synset':
        'cookie.n.01',
    'synonyms': ['cookie', 'cooky', 'biscuit_(cookie)'],
    'def':
        "any of various small flat sweet cakes (`biscuit' is the British term)",
    'name':
        'cookie'
}, {
    'frequency': 'r',
    'id': 306,
    'synset': 'cookie_jar.n.01',
    'synonyms': ['cookie_jar', 'cooky_jar'],
    'def': 'a jar in which cookies are kept (and sometimes money is hidden)',
    'name': 'cookie_jar'
}, {
    'frequency':
        'r',
    'id':
        307,
    'synset':
        'cooking_utensil.n.01',
    'synonyms': ['cooking_utensil'],
    'def':
        'a kitchen utensil made of material that does not melt easily; used for cooking',
    'name':
        'cooking_utensil'
}, {
    'frequency': 'f',
    'id': 308,
    'synset': 'cooler.n.01',
    'synonyms': ['cooler_(for_food)', 'ice_chest'],
    'def': 'an insulated box for storing food often with ice',
    'name': 'cooler_(for_food)'
}, {
    'frequency': 'c',
    'id': 309,
    'synset': 'cork.n.04',
    'synonyms': ['cork_(bottle_plug)', 'bottle_cork'],
    'def': 'the plug in the mouth of a bottle (especially a wine bottle)',
    'name': 'cork_(bottle_plug)'
}, {
    'frequency': 'r',
    'id': 310,
    'synset': 'corkboard.n.01',
    'synonyms': ['corkboard'],
    'def': 'a sheet consisting of cork granules',
    'name': 'corkboard'
}, {
    'frequency': 'r',
    'id': 311,
    'synset': 'corkscrew.n.01',
    'synonyms': ['corkscrew', 'bottle_screw'],
    'def': 'a bottle opener that pulls corks',
    'name': 'corkscrew'
}, {
    'frequency': 'c',
    'id': 312,
    'synset': 'corn.n.03',
    'synonyms': ['edible_corn', 'corn', 'maize'],
    'def': 'ears of corn that can be prepared and served for human food',
    'name': 'edible_corn'
}, {
    'frequency': 'r',
    'id': 313,
    'synset': 'cornbread.n.01',
    'synonyms': ['cornbread'],
    'def': 'bread made primarily of cornmeal',
    'name': 'cornbread'
}, {
    'frequency':
        'c',
    'id':
        314,
    'synset':
        'cornet.n.01',
    'synonyms': ['cornet', 'horn', 'trumpet'],
    'def':
        'a brass musical instrument with a narrow tube and a flared bell and many valves',
    'name':
        'cornet'
}, {
    'frequency':
        'c',
    'id':
        315,
    'synset':
        'cornice.n.01',
    'synonyms': ['cornice', 'valance', 'valance_board', 'pelmet'],
    'def':
        'a decorative framework to conceal curtain fixtures at the top of a window casing',
    'name':
        'cornice'
}, {
    'frequency': 'r',
    'id': 316,
    'synset': 'cornmeal.n.01',
    'synonyms': ['cornmeal'],
    'def': 'coarsely ground corn',
    'name': 'cornmeal'
}, {
    'frequency': 'r',
    'id': 317,
    'synset': 'corset.n.01',
    'synonyms': ['corset', 'girdle'],
    'def': "a woman's close-fitting foundation garment",
    'name': 'corset'
}, {
    'frequency':
        'r',
    'id':
        318,
    'synset':
        'cos.n.02',
    'synonyms': ['romaine_lettuce'],
    'def':
        'lettuce with long dark-green leaves in a loosely packed elongated head',
    'name':
        'romaine_lettuce'
}, {
    'frequency': 'c',
    'id': 319,
    'synset': 'costume.n.04',
    'synonyms': ['costume'],
    'def': 'the attire characteristic of a country or a time or a social class',
    'name': 'costume'
}, {
    'frequency': 'r',
    'id': 320,
    'synset': 'cougar.n.01',
    'synonyms': ['cougar', 'puma', 'catamount', 'mountain_lion', 'panther'],
    'def': 'large American feline resembling a lion',
    'name': 'cougar'
}, {
    'frequency':
        'r',
    'id':
        321,
    'synset':
        'coverall.n.01',
    'synonyms': ['coverall'],
    'def':
        'a loose-fitting protective garment that is worn over other clothing',
    'name':
        'coverall'
}, {
    'frequency':
        'r',
    'id':
        322,
    'synset':
        'cowbell.n.01',
    'synonyms': ['cowbell'],
    'def':
        'a bell hung around the neck of cow so that the cow can be easily located',
    'name':
        'cowbell'
}, {
    'frequency':
        'f',
    'id':
        323,
    'synset':
        'cowboy_hat.n.01',
    'synonyms': ['cowboy_hat', 'ten-gallon_hat'],
    'def':
        'a hat with a wide brim and a soft crown; worn by American ranch hands',
    'name':
        'cowboy_hat'
}, {
    'frequency':
        'r',
    'id':
        324,
    'synset':
        'crab.n.01',
    'synonyms': ['crab_(animal)'],
    'def':
        'decapod having eyes on short stalks and a broad flattened shell and pincers',
    'name':
        'crab_(animal)'
}, {
    'frequency': 'c',
    'id': 325,
    'synset': 'cracker.n.01',
    'synonyms': ['cracker'],
    'def': 'a thin crisp wafer',
    'name': 'cracker'
}, {
    'frequency': 'r',
    'id': 326,
    'synset': 'crape.n.01',
    'synonyms': ['crape', 'crepe', 'French_pancake'],
    'def': 'small very thin pancake',
    'name': 'crape'
}, {
    'frequency': 'f',
    'id': 327,
    'synset': 'crate.n.01',
    'synonyms': ['crate'],
    'def': 'a rugged box (usually made of wood); used for shipping',
    'name': 'crate'
}, {
    'frequency':
        'r',
    'id':
        328,
    'synset':
        'crayon.n.01',
    'synonyms': ['crayon', 'wax_crayon'],
    'def':
        'writing or drawing implement made of a colored stick of composition wax',
    'name':
        'crayon'
}, {
    'frequency': 'r',
    'id': 329,
    'synset': 'cream_pitcher.n.01',
    'synonyms': ['cream_pitcher'],
    'def': 'a small pitcher for serving cream',
    'name': 'cream_pitcher'
}, {
    'frequency': 'r',
    'id': 330,
    'synset': 'credit_card.n.01',
    'synonyms': ['credit_card', 'charge_card', 'debit_card'],
    'def': 'a card, usually plastic, used to pay for goods and services',
    'name': 'credit_card'
}, {
    'frequency': 'c',
    'id': 331,
    'synset': 'crescent_roll.n.01',
    'synonyms': ['crescent_roll', 'croissant'],
    'def': 'very rich flaky crescent-shaped roll',
    'name': 'crescent_roll'
}, {
    'frequency': 'c',
    'id': 332,
    'synset': 'crib.n.01',
    'synonyms': ['crib', 'cot'],
    'def': 'baby bed with high sides made of slats',
    'name': 'crib'
}, {
    'frequency': 'c',
    'id': 333,
    'synset': 'crock.n.03',
    'synonyms': ['crock_pot', 'earthenware_jar'],
    'def': 'an earthen jar (made of baked clay)',
    'name': 'crock_pot'
}, {
    'frequency': 'f',
    'id': 334,
    'synset': 'crossbar.n.01',
    'synonyms': ['crossbar'],
    'def': 'a horizontal bar that goes across something',
    'name': 'crossbar'
}, {
    'frequency': 'r',
    'id': 335,
    'synset': 'crouton.n.01',
    'synonyms': ['crouton'],
    'def': 'a small piece of toasted or fried bread; served in soup or salads',
    'name': 'crouton'
}, {
    'frequency': 'r',
    'id': 336,
    'synset': 'crow.n.01',
    'synonyms': ['crow'],
    'def': 'black birds having a raucous call',
    'name': 'crow'
}, {
    'frequency': 'c',
    'id': 337,
    'synset': 'crown.n.04',
    'synonyms': ['crown'],
    'def': 'an ornamental jeweled headdress signifying sovereignty',
    'name': 'crown'
}, {
    'frequency': 'c',
    'id': 338,
    'synset': 'crucifix.n.01',
    'synonyms': ['crucifix'],
    'def': 'representation of the cross on which Jesus died',
    'name': 'crucifix'
}, {
    'frequency': 'c',
    'id': 339,
    'synset': 'cruise_ship.n.01',
    'synonyms': ['cruise_ship', 'cruise_liner'],
    'def': 'a passenger ship used commercially for pleasure cruises',
    'name': 'cruise_ship'
}, {
    'frequency': 'c',
    'id': 340,
    'synset': 'cruiser.n.01',
    'synonyms': ['police_cruiser', 'patrol_car', 'police_car', 'squad_car'],
    'def': 'a car in which policemen cruise the streets',
    'name': 'police_cruiser'
}, {
    'frequency': 'c',
    'id': 341,
    'synset': 'crumb.n.03',
    'synonyms': ['crumb'],
    'def': 'small piece of e.g. bread or cake',
    'name': 'crumb'
}, {
    'frequency':
        'r',
    'id':
        342,
    'synset':
        'crutch.n.01',
    'synonyms': ['crutch'],
    'def':
        'a wooden or metal staff that fits under the armpit and reaches to the ground',
    'name':
        'crutch'
}, {
    'frequency':
        'c',
    'id':
        343,
    'synset':
        'cub.n.03',
    'synonyms': ['cub_(animal)'],
    'def':
        'the young of certain carnivorous mammals such as the bear or wolf or lion',
    'name':
        'cub_(animal)'
}, {
    'frequency': 'r',
    'id': 344,
    'synset': 'cube.n.05',
    'synonyms': ['cube', 'square_block'],
    'def': 'a block in the (approximate) shape of a cube',
    'name': 'cube'
}, {
    'frequency':
        'f',
    'id':
        345,
    'synset':
        'cucumber.n.02',
    'synonyms': ['cucumber', 'cuke'],
    'def':
        'cylindrical green fruit with thin green rind and white flesh eaten as a vegetable',
    'name':
        'cucumber'
}, {
    'frequency':
        'c',
    'id':
        346,
    'synset':
        'cufflink.n.01',
    'synonyms': ['cufflink'],
    'def':
        'jewelry consisting of linked buttons used to fasten the cuffs of a shirt',
    'name':
        'cufflink'
}, {
    'frequency':
        'f',
    'id':
        347,
    'synset':
        'cup.n.01',
    'synonyms': ['cup'],
    'def':
        'a small open container usually used for drinking; usually has a handle',
    'name':
        'cup'
}, {
    'frequency':
        'c',
    'id':
        348,
    'synset':
        'cup.n.08',
    'synonyms': ['trophy_cup'],
    'def':
        'a metal vessel with handles that is awarded as a trophy to a competition winner',
    'name':
        'trophy_cup'
}, {
    'frequency': 'c',
    'id': 349,
    'synset': 'cupcake.n.01',
    'synonyms': ['cupcake'],
    'def': 'small cake baked in a muffin tin',
    'name': 'cupcake'
}, {
    'frequency': 'r',
    'id': 350,
    'synset': 'curler.n.01',
    'synonyms': ['hair_curler', 'hair_roller', 'hair_crimper'],
    'def': 'a cylindrical tube around which the hair is wound to curl it',
    'name': 'hair_curler'
}, {
    'frequency':
        'r',
    'id':
        351,
    'synset':
        'curling_iron.n.01',
    'synonyms': ['curling_iron'],
    'def':
        'a cylindrical home appliance that heats hair that has been curled around it',
    'name':
        'curling_iron'
}, {
    'frequency': 'f',
    'id': 352,
    'synset': 'curtain.n.01',
    'synonyms': ['curtain', 'drapery'],
    'def': 'hanging cloth used as a blind (especially for a window)',
    'name': 'curtain'
}, {
    'frequency':
        'f',
    'id':
        353,
    'synset':
        'cushion.n.03',
    'synonyms': ['cushion'],
    'def':
        'a soft bag filled with air or padding such as feathers or foam rubber',
    'name':
        'cushion'
}, {
    'frequency': 'r',
    'id': 354,
    'synset': 'custard.n.01',
    'synonyms': ['custard'],
    'def': 'sweetened mixture of milk and eggs baked or boiled or frozen',
    'name': 'custard'
}, {
    'frequency': 'c',
    'id': 355,
    'synset': 'cutter.n.06',
    'synonyms': ['cutting_tool'],
    'def': 'a cutting implement; a tool for cutting',
    'name': 'cutting_tool'
}, {
    'frequency': 'r',
    'id': 356,
    'synset': 'cylinder.n.04',
    'synonyms': ['cylinder'],
    'def': 'a cylindrical container',
    'name': 'cylinder'
}, {
    'frequency': 'r',
    'id': 357,
    'synset': 'cymbal.n.01',
    'synonyms': ['cymbal'],
    'def': 'a percussion instrument consisting of a concave brass disk',
    'name': 'cymbal'
}, {
    'frequency':
        'r',
    'id':
        358,
    'synset':
        'dachshund.n.01',
    'synonyms': ['dachshund', 'dachsie', 'badger_dog'],
    'def':
        'small long-bodied short-legged breed of dog having a short sleek coat and long drooping ears',
    'name':
        'dachshund'
}, {
    'frequency': 'r',
    'id': 359,
    'synset': 'dagger.n.01',
    'synonyms': ['dagger'],
    'def': 'a short knife with a pointed blade used for piercing or stabbing',
    'name': 'dagger'
}, {
    'frequency':
        'r',
    'id':
        360,
    'synset':
        'dartboard.n.01',
    'synonyms': ['dartboard'],
    'def':
        'a circular board of wood or cork used as the target in the game of darts',
    'name':
        'dartboard'
}, {
    'frequency': 'r',
    'id': 361,
    'synset': 'date.n.08',
    'synonyms': ['date_(fruit)'],
    'def': 'sweet edible fruit of the date palm with a single long woody seed',
    'name': 'date_(fruit)'
}, {
    'frequency':
        'f',
    'id':
        362,
    'synset':
        'deck_chair.n.01',
    'synonyms': ['deck_chair', 'beach_chair'],
    'def':
        'a folding chair for use outdoors; a wooden frame supports a length of canvas',
    'name':
        'deck_chair'
}, {
    'frequency':
        'c',
    'id':
        363,
    'synset':
        'deer.n.01',
    'synonyms': ['deer', 'cervid'],
    'def':
        "distinguished from Bovidae by the male's having solid deciduous antlers",
    'name':
        'deer'
}, {
    'frequency': 'c',
    'id': 364,
    'synset': 'dental_floss.n.01',
    'synonyms': ['dental_floss', 'floss'],
    'def': 'a soft thread for cleaning the spaces between the teeth',
    'name': 'dental_floss'
}, {
    'frequency':
        'f',
    'id':
        365,
    'synset':
        'desk.n.01',
    'synonyms': ['desk'],
    'def':
        'a piece of furniture with a writing surface and usually drawers or other compartments',
    'name':
        'desk'
}, {
    'frequency': 'r',
    'id': 366,
    'synset': 'detergent.n.01',
    'synonyms': ['detergent'],
    'def': 'a surface-active chemical widely used in industry and laundering',
    'name': 'detergent'
}, {
    'frequency':
        'c',
    'id':
        367,
    'synset':
        'diaper.n.01',
    'synonyms': ['diaper'],
    'def':
        'garment consisting of a folded cloth drawn up between the legs and fastened at the waist',
    'name':
        'diaper'
}, {
    'frequency':
        'r',
    'id':
        368,
    'synset':
        'diary.n.01',
    'synonyms': ['diary', 'journal'],
    'def':
        'a daily written record of (usually personal) experiences and observations',
    'name':
        'diary'
}, {
    'frequency': 'r',
    'id': 369,
    'synset': 'die.n.01',
    'synonyms': ['die', 'dice'],
    'def': 'a small cube with 1 to 6 spots on the six faces; used in gambling',
    'name': 'die'
}, {
    'frequency':
        'r',
    'id':
        370,
    'synset':
        'dinghy.n.01',
    'synonyms': ['dinghy', 'dory', 'rowboat'],
    'def':
        'a small boat of shallow draft with seats and oars with which it is propelled',
    'name':
        'dinghy'
}, {
    'frequency': 'f',
    'id': 371,
    'synset': 'dining_table.n.01',
    'synonyms': ['dining_table'],
    'def': 'a table at which meals are served',
    'name': 'dining_table'
}, {
    'frequency': 'r',
    'id': 372,
    'synset': 'dinner_jacket.n.01',
    'synonyms': ['tux', 'tuxedo'],
    'def': 'semiformal evening dress for men',
    'name': 'tux'
}, {
    'frequency':
        'c',
    'id':
        373,
    'synset':
        'dish.n.01',
    'synonyms': ['dish'],
    'def':
        'a piece of dishware normally used as a container for holding or serving food',
    'name':
        'dish'
}, {
    'frequency': 'c',
    'id': 374,
    'synset': 'dish.n.05',
    'synonyms': ['dish_antenna'],
    'def': 'directional antenna consisting of a parabolic reflector',
    'name': 'dish_antenna'
}, {
    'frequency': 'c',
    'id': 375,
    'synset': 'dishrag.n.01',
    'synonyms': ['dishrag', 'dishcloth'],
    'def': 'a cloth for washing dishes',
    'name': 'dishrag'
}, {
    'frequency': 'c',
    'id': 376,
    'synset': 'dishtowel.n.01',
    'synonyms': ['dishtowel', 'tea_towel'],
    'def': 'a towel for drying dishes',
    'name': 'dishtowel'
}, {
    'frequency': 'f',
    'id': 377,
    'synset': 'dishwasher.n.01',
    'synonyms': ['dishwasher', 'dishwashing_machine'],
    'def': 'a machine for washing dishes',
    'name': 'dishwasher'
}, {
    'frequency': 'r',
    'id': 378,
    'synset': 'dishwasher_detergent.n.01',
    'synonyms': [
        'dishwasher_detergent', 'dishwashing_detergent', 'dishwashing_liquid'
    ],
    'def': 'a low-sudsing detergent designed for use in dishwashers',
    'name': 'dishwasher_detergent'
}, {
    'frequency':
        'r',
    'id':
        379,
    'synset':
        'diskette.n.01',
    'synonyms': ['diskette', 'floppy', 'floppy_disk'],
    'def':
        'a small plastic magnetic disk enclosed in a stiff envelope used to store data',
    'name':
        'diskette'
}, {
    'frequency':
        'c',
    'id':
        380,
    'synset':
        'dispenser.n.01',
    'synonyms': ['dispenser'],
    'def':
        'a container so designed that the contents can be used in prescribed amounts',
    'name':
        'dispenser'
}, {
    'frequency': 'c',
    'id': 381,
    'synset': 'dixie_cup.n.01',
    'synonyms': ['Dixie_cup', 'paper_cup'],
    'def': 'a disposable cup made of paper; for holding drinks',
    'name': 'Dixie_cup'
}, {
    'frequency': 'f',
    'id': 382,
    'synset': 'dog.n.01',
    'synonyms': ['dog'],
    'def': 'a common domesticated dog',
    'name': 'dog'
}, {
    'frequency': 'f',
    'id': 383,
    'synset': 'dog_collar.n.01',
    'synonyms': ['dog_collar'],
    'def': 'a collar for a dog',
    'name': 'dog_collar'
}, {
    'frequency': 'c',
    'id': 384,
    'synset': 'doll.n.01',
    'synonyms': ['doll'],
    'def': 'a toy replica of a HUMAN (NOT AN ANIMAL)',
    'name': 'doll'
}, {
    'frequency': 'r',
    'id': 385,
    'synset': 'dollar.n.02',
    'synonyms': ['dollar', 'dollar_bill', 'one_dollar_bill'],
    'def': 'a piece of paper money worth one dollar',
    'name': 'dollar'
}, {
    'frequency':
        'r',
    'id':
        386,
    'synset':
        'dolphin.n.02',
    'synonyms': ['dolphin'],
    'def':
        'any of various small toothed whales with a beaklike snout; larger than porpoises',
    'name':
        'dolphin'
}, {
    'frequency':
        'c',
    'id':
        387,
    'synset':
        'domestic_ass.n.01',
    'synonyms': ['domestic_ass', 'donkey'],
    'def':
        'domestic beast of burden descended from the African wild ass; patient but stubborn',
    'name':
        'domestic_ass'
}, {
    'frequency':
        'r',
    'id':
        388,
    'synset':
        'domino.n.03',
    'synonyms': ['eye_mask'],
    'def':
        'a mask covering the upper part of the face but with holes for the eyes',
    'name':
        'eye_mask'
}, {
    'frequency':
        'r',
    'id':
        389,
    'synset':
        'doorbell.n.01',
    'synonyms': ['doorbell', 'buzzer'],
    'def':
        'a button at an outer door that gives a ringing or buzzing signal when pushed',
    'name':
        'doorbell'
}, {
    'frequency':
        'f',
    'id':
        390,
    'synset':
        'doorknob.n.01',
    'synonyms': ['doorknob', 'doorhandle'],
    'def':
        "a knob used to open a door (often called `doorhandle' in Great Britain)",
    'name':
        'doorknob'
}, {
    'frequency':
        'c',
    'id':
        391,
    'synset':
        'doormat.n.02',
    'synonyms': ['doormat', 'welcome_mat'],
    'def':
        'a mat placed outside an exterior door for wiping the shoes before entering',
    'name':
        'doormat'
}, {
    'frequency': 'f',
    'id': 392,
    'synset': 'doughnut.n.02',
    'synonyms': ['doughnut', 'donut'],
    'def': 'a small ring-shaped friedcake',
    'name': 'doughnut'
}, {
    'frequency': 'r',
    'id': 393,
    'synset': 'dove.n.01',
    'synonyms': ['dove'],
    'def': 'any of numerous small pigeons',
    'name': 'dove'
}, {
    'frequency':
        'r',
    'id':
        394,
    'synset':
        'dragonfly.n.01',
    'synonyms': ['dragonfly'],
    'def':
        'slender-bodied non-stinging insect having iridescent wings that are outspread at rest',
    'name':
        'dragonfly'
}, {
    'frequency':
        'f',
    'id':
        395,
    'synset':
        'drawer.n.01',
    'synonyms': ['drawer'],
    'def':
        'a boxlike container in a piece of furniture; made so as to slide in and out',
    'name':
        'drawer'
}, {
    'frequency': 'c',
    'id': 396,
    'synset': 'drawers.n.01',
    'synonyms': ['underdrawers', 'boxers', 'boxershorts'],
    'def': 'underpants worn by men',
    'name': 'underdrawers'
}, {
    'frequency': 'f',
    'id': 397,
    'synset': 'dress.n.01',
    'synonyms': ['dress', 'frock'],
    'def': 'a one-piece garment for a woman; has skirt and bodice',
    'name': 'dress'
}, {
    'frequency':
        'c',
    'id':
        398,
    'synset':
        'dress_hat.n.01',
    'synonyms': ['dress_hat', 'high_hat', 'opera_hat', 'silk_hat', 'top_hat'],
    'def':
        "a man's hat with a tall crown; usually covered with silk or with beaver fur",
    'name':
        'dress_hat'
}, {
    'frequency': 'c',
    'id': 399,
    'synset': 'dress_suit.n.01',
    'synonyms': ['dress_suit'],
    'def': 'formalwear consisting of full evening dress for men',
    'name': 'dress_suit'
}, {
    'frequency': 'c',
    'id': 400,
    'synset': 'dresser.n.05',
    'synonyms': ['dresser'],
    'def': 'a cabinet with shelves',
    'name': 'dresser'
}, {
    'frequency':
        'c',
    'id':
        401,
    'synset':
        'drill.n.01',
    'synonyms': ['drill'],
    'def':
        'a tool with a sharp rotating point for making holes in hard materials',
    'name':
        'drill'
}, {
    'frequency': 'r',
    'id': 402,
    'synset': 'drinking_fountain.n.01',
    'synonyms': ['drinking_fountain'],
    'def': 'a public fountain to provide a jet of drinking water',
    'name': 'drinking_fountain'
}, {
    'frequency': 'r',
    'id': 403,
    'synset': 'drone.n.04',
    'synonyms': ['drone'],
    'def': 'an aircraft without a pilot that is operated by remote control',
    'name': 'drone'
}, {
    'frequency':
        'r',
    'id':
        404,
    'synset':
        'dropper.n.01',
    'synonyms': ['dropper', 'eye_dropper'],
    'def':
        'pipet consisting of a small tube with a vacuum bulb at one end for drawing liquid in and releasing it a drop at a time',
    'name':
        'dropper'
}, {
    'frequency':
        'c',
    'id':
        405,
    'synset':
        'drum.n.01',
    'synonyms': ['drum_(musical_instrument)'],
    'def':
        'a musical percussion instrument; usually consists of a hollow cylinder with a membrane stretched across each end',
    'name':
        'drum_(musical_instrument)'
}, {
    'frequency': 'r',
    'id': 406,
    'synset': 'drumstick.n.02',
    'synonyms': ['drumstick'],
    'def': 'a stick used for playing a drum',
    'name': 'drumstick'
}, {
    'frequency': 'f',
    'id': 407,
    'synset': 'duck.n.01',
    'synonyms': ['duck'],
    'def': 'small web-footed broad-billed swimming bird',
    'name': 'duck'
}, {
    'frequency': 'r',
    'id': 408,
    'synset': 'duckling.n.02',
    'synonyms': ['duckling'],
    'def': 'young duck',
    'name': 'duckling'
}, {
    'frequency': 'c',
    'id': 409,
    'synset': 'duct_tape.n.01',
    'synonyms': ['duct_tape'],
    'def': 'a wide silvery adhesive tape',
    'name': 'duct_tape'
}, {
    'frequency': 'f',
    'id': 410,
    'synset': 'duffel_bag.n.01',
    'synonyms': ['duffel_bag', 'duffle_bag', 'duffel', 'duffle'],
    'def': 'a large cylindrical bag of heavy cloth',
    'name': 'duffel_bag'
}, {
    'frequency':
        'r',
    'id':
        411,
    'synset':
        'dumbbell.n.01',
    'synonyms': ['dumbbell'],
    'def':
        'an exercising weight with two ball-like ends connected by a short handle',
    'name':
        'dumbbell'
}, {
    'frequency': 'c',
    'id': 412,
    'synset': 'dumpster.n.01',
    'synonyms': ['dumpster'],
    'def': 'a container designed to receive and transport and dump waste',
    'name': 'dumpster'
}, {
    'frequency': 'r',
    'id': 413,
    'synset': 'dustpan.n.02',
    'synonyms': ['dustpan'],
    'def': 'a short-handled receptacle into which dust can be swept',
    'name': 'dustpan'
}, {
    'frequency': 'r',
    'id': 414,
    'synset': 'dutch_oven.n.02',
    'synonyms': ['Dutch_oven'],
    'def': 'iron or earthenware cooking pot; used for stews',
    'name': 'Dutch_oven'
}, {
    'frequency':
        'c',
    'id':
        415,
    'synset':
        'eagle.n.01',
    'synonyms': ['eagle'],
    'def':
        'large birds of prey noted for their broad wings and strong soaring flight',
    'name':
        'eagle'
}, {
    'frequency':
        'f',
    'id':
        416,
    'synset':
        'earphone.n.01',
    'synonyms': ['earphone', 'earpiece', 'headphone'],
    'def':
        'device for listening to audio that is held over or inserted into the ear',
    'name':
        'earphone'
}, {
    'frequency': 'r',
    'id': 417,
    'synset': 'earplug.n.01',
    'synonyms': ['earplug'],
    'def': 'a soft plug that is inserted into the ear canal to block sound',
    'name': 'earplug'
}, {
    'frequency': 'f',
    'id': 418,
    'synset': 'earring.n.01',
    'synonyms': ['earring'],
    'def': 'jewelry to ornament the ear',
    'name': 'earring'
}, {
    'frequency':
        'c',
    'id':
        419,
    'synset':
        'easel.n.01',
    'synonyms': ['easel'],
    'def':
        "an upright tripod for displaying something (usually an artist's canvas)",
    'name':
        'easel'
}, {
    'frequency': 'r',
    'id': 420,
    'synset': 'eclair.n.01',
    'synonyms': ['eclair'],
    'def': 'oblong cream puff',
    'name': 'eclair'
}, {
    'frequency': 'r',
    'id': 421,
    'synset': 'eel.n.01',
    'synonyms': ['eel'],
    'def': 'an elongate fish with fatty flesh',
    'name': 'eel'
}, {
    'frequency': 'f',
    'id': 422,
    'synset': 'egg.n.02',
    'synonyms': ['egg', 'eggs'],
    'def': 'oval reproductive body of a fowl (especially a hen) used as food',
    'name': 'egg'
}, {
    'frequency': 'r',
    'id': 423,
    'synset': 'egg_roll.n.01',
    'synonyms': ['egg_roll', 'spring_roll'],
    'def': 'minced vegetables and meat wrapped in a pancake and fried',
    'name': 'egg_roll'
}, {
    'frequency': 'c',
    'id': 424,
    'synset': 'egg_yolk.n.01',
    'synonyms': ['egg_yolk', 'yolk_(egg)'],
    'def': 'the yellow spherical part of an egg',
    'name': 'egg_yolk'
}, {
    'frequency': 'c',
    'id': 425,
    'synset': 'eggbeater.n.02',
    'synonyms': ['eggbeater', 'eggwhisk'],
    'def': 'a mixer for beating eggs or whipping cream',
    'name': 'eggbeater'
}, {
    'frequency': 'c',
    'id': 426,
    'synset': 'eggplant.n.01',
    'synonyms': ['eggplant', 'aubergine'],
    'def': 'egg-shaped vegetable having a shiny skin typically dark purple',
    'name': 'eggplant'
}, {
    'frequency': 'r',
    'id': 427,
    'synset': 'electric_chair.n.01',
    'synonyms': ['electric_chair'],
    'def': 'a chair-shaped instrument of execution by electrocution',
    'name': 'electric_chair'
}, {
    'frequency':
        'f',
    'id':
        428,
    'synset':
        'electric_refrigerator.n.01',
    'synonyms': ['refrigerator'],
    'def':
        'a refrigerator in which the coolant is pumped around by an electric motor',
    'name':
        'refrigerator'
}, {
    'frequency': 'f',
    'id': 429,
    'synset': 'elephant.n.01',
    'synonyms': ['elephant'],
    'def': 'a common elephant',
    'name': 'elephant'
}, {
    'frequency': 'r',
    'id': 430,
    'synset': 'elk.n.01',
    'synonyms': ['elk', 'moose'],
    'def': 'large northern deer with enormous flattened antlers in the male',
    'name': 'elk'
}, {
    'frequency':
        'c',
    'id':
        431,
    'synset':
        'envelope.n.01',
    'synonyms': ['envelope'],
    'def':
        'a flat (usually rectangular) container for a letter, thin package, etc.',
    'name':
        'envelope'
}, {
    'frequency': 'c',
    'id': 432,
    'synset': 'eraser.n.01',
    'synonyms': ['eraser'],
    'def': 'an implement used to erase something',
    'name': 'eraser'
}, {
    'frequency':
        'r',
    'id':
        433,
    'synset':
        'escargot.n.01',
    'synonyms': ['escargot'],
    'def':
        'edible snail usually served in the shell with a sauce of melted butter and garlic',
    'name':
        'escargot'
}, {
    'frequency': 'r',
    'id': 434,
    'synset': 'eyepatch.n.01',
    'synonyms': ['eyepatch'],
    'def': 'a protective cloth covering for an injured eye',
    'name': 'eyepatch'
}, {
    'frequency':
        'r',
    'id':
        435,
    'synset':
        'falcon.n.01',
    'synonyms': ['falcon'],
    'def':
        'birds of prey having long pointed powerful wings adapted for swift flight',
    'name':
        'falcon'
}, {
    'frequency':
        'f',
    'id':
        436,
    'synset':
        'fan.n.01',
    'synonyms': ['fan'],
    'def':
        'a device for creating a current of air by movement of a surface or surfaces',
    'name':
        'fan'
}, {
    'frequency': 'f',
    'id': 437,
    'synset': 'faucet.n.01',
    'synonyms': ['faucet', 'spigot', 'tap'],
    'def': 'a regulator for controlling the flow of a liquid from a reservoir',
    'name': 'faucet'
}, {
    'frequency': 'r',
    'id': 438,
    'synset': 'fedora.n.01',
    'synonyms': ['fedora'],
    'def': 'a hat made of felt with a creased crown',
    'name': 'fedora'
}, {
    'frequency':
        'r',
    'id':
        439,
    'synset':
        'ferret.n.02',
    'synonyms': ['ferret'],
    'def':
        'domesticated albino variety of the European polecat bred for hunting rats and rabbits',
    'name':
        'ferret'
}, {
    'frequency':
        'c',
    'id':
        440,
    'synset':
        'ferris_wheel.n.01',
    'synonyms': ['Ferris_wheel'],
    'def':
        'a large wheel with suspended seats that remain upright as the wheel rotates',
    'name':
        'Ferris_wheel'
}, {
    'frequency':
        'r',
    'id':
        441,
    'synset':
        'ferry.n.01',
    'synonyms': ['ferry', 'ferryboat'],
    'def':
        'a boat that transports people or vehicles across a body of water and operates on a regular schedule',
    'name':
        'ferry'
}, {
    'frequency':
        'r',
    'id':
        442,
    'synset':
        'fig.n.04',
    'synonyms': ['fig_(fruit)'],
    'def':
        'fleshy sweet pear-shaped yellowish or purple fruit eaten fresh or preserved or dried',
    'name':
        'fig_(fruit)'
}, {
    'frequency':
        'c',
    'id':
        443,
    'synset':
        'fighter.n.02',
    'synonyms': ['fighter_jet', 'fighter_aircraft', 'attack_aircraft'],
    'def':
        'a high-speed military or naval airplane designed to destroy enemy targets',
    'name':
        'fighter_jet'
}, {
    'frequency': 'f',
    'id': 444,
    'synset': 'figurine.n.01',
    'synonyms': ['figurine'],
    'def': 'a small carved or molded figure',
    'name': 'figurine'
}, {
    'frequency':
        'c',
    'id':
        445,
    'synset':
        'file.n.03',
    'synonyms': ['file_cabinet', 'filing_cabinet'],
    'def':
        'office furniture consisting of a container for keeping papers in order',
    'name':
        'file_cabinet'
}, {
    'frequency':
        'r',
    'id':
        446,
    'synset':
        'file.n.04',
    'synonyms': ['file_(tool)'],
    'def':
        'a steel hand tool with small sharp teeth on some or all of its surfaces; used for smoothing wood or metal',
    'name':
        'file_(tool)'
}, {
    'frequency': 'f',
    'id': 447,
    'synset': 'fire_alarm.n.02',
    'synonyms': ['fire_alarm', 'smoke_alarm'],
    'def': 'an alarm that is tripped off by fire or smoke',
    'name': 'fire_alarm'
}, {
    'frequency':
        'c',
    'id':
        448,
    'synset':
        'fire_engine.n.01',
    'synonyms': ['fire_engine', 'fire_truck'],
    'def':
        'large trucks that carry firefighters and equipment to the site of a fire',
    'name':
        'fire_engine'
}, {
    'frequency': 'c',
    'id': 449,
    'synset': 'fire_extinguisher.n.01',
    'synonyms': ['fire_extinguisher', 'extinguisher'],
    'def': 'a manually operated device for extinguishing small fires',
    'name': 'fire_extinguisher'
}, {
    'frequency':
        'c',
    'id':
        450,
    'synset':
        'fire_hose.n.01',
    'synonyms': ['fire_hose'],
    'def':
        'a large hose that carries water from a fire hydrant to the site of the fire',
    'name':
        'fire_hose'
}, {
    'frequency':
        'f',
    'id':
        451,
    'synset':
        'fireplace.n.01',
    'synonyms': ['fireplace'],
    'def':
        'an open recess in a wall at the base of a chimney where a fire can be built',
    'name':
        'fireplace'
}, {
    'frequency': 'f',
    'id': 452,
    'synset': 'fireplug.n.01',
    'synonyms': ['fireplug', 'fire_hydrant', 'hydrant'],
    'def': 'an upright hydrant for drawing water to use in fighting a fire',
    'name': 'fireplug'
}, {
    'frequency':
        'c',
    'id':
        453,
    'synset':
        'fish.n.01',
    'synonyms': ['fish'],
    'def':
        'any of various mostly cold-blooded aquatic vertebrates usually having scales and breathing through gills',
    'name':
        'fish'
}, {
    'frequency': 'r',
    'id': 454,
    'synset': 'fish.n.02',
    'synonyms': ['fish_(food)'],
    'def': 'the flesh of fish used as food',
    'name': 'fish_(food)'
}, {
    'frequency': 'r',
    'id': 455,
    'synset': 'fishbowl.n.02',
    'synonyms': ['fishbowl', 'goldfish_bowl'],
    'def': 'a transparent bowl in which small fish are kept',
    'name': 'fishbowl'
}, {
    'frequency': 'r',
    'id': 456,
    'synset': 'fishing_boat.n.01',
    'synonyms': ['fishing_boat', 'fishing_vessel'],
    'def': 'a vessel for fishing',
    'name': 'fishing_boat'
}, {
    'frequency': 'c',
    'id': 457,
    'synset': 'fishing_rod.n.01',
    'synonyms': ['fishing_rod', 'fishing_pole'],
    'def': 'a rod that is used in fishing to extend the fishing line',
    'name': 'fishing_rod'
}, {
    'frequency':
        'f',
    'id':
        458,
    'synset':
        'flag.n.01',
    'synonyms': ['flag'],
    'def':
        'emblem usually consisting of a rectangular piece of cloth of distinctive design (do not include pole)',
    'name':
        'flag'
}, {
    'frequency': 'f',
    'id': 459,
    'synset': 'flagpole.n.02',
    'synonyms': ['flagpole', 'flagstaff'],
    'def': 'a tall staff or pole on which a flag is raised',
    'name': 'flagpole'
}, {
    'frequency': 'c',
    'id': 460,
    'synset': 'flamingo.n.01',
    'synonyms': ['flamingo'],
    'def': 'large pink web-footed bird with down-bent bill',
    'name': 'flamingo'
}, {
    'frequency': 'c',
    'id': 461,
    'synset': 'flannel.n.01',
    'synonyms': ['flannel'],
    'def': 'a soft light woolen fabric; used for clothing',
    'name': 'flannel'
}, {
    'frequency': 'r',
    'id': 462,
    'synset': 'flash.n.10',
    'synonyms': ['flash', 'flashbulb'],
    'def': 'a lamp for providing momentary light to take a photograph',
    'name': 'flash'
}, {
    'frequency': 'c',
    'id': 463,
    'synset': 'flashlight.n.01',
    'synonyms': ['flashlight', 'torch'],
    'def': 'a small portable battery-powered electric lamp',
    'name': 'flashlight'
}, {
    'frequency': 'r',
    'id': 464,
    'synset': 'fleece.n.03',
    'synonyms': ['fleece'],
    'def': 'a soft bulky fabric with deep pile; used chiefly for clothing',
    'name': 'fleece'
}, {
    'frequency': 'f',
    'id': 465,
    'synset': 'flip-flop.n.02',
    'synonyms': ['flip-flop_(sandal)'],
    'def': 'a backless sandal held to the foot by a thong between two toes',
    'name': 'flip-flop_(sandal)'
}, {
    'frequency': 'c',
    'id': 466,
    'synset': 'flipper.n.01',
    'synonyms': ['flipper_(footwear)', 'fin_(footwear)'],
    'def': 'a shoe to aid a person in swimming',
    'name': 'flipper_(footwear)'
}, {
    'frequency': 'f',
    'id': 467,
    'synset': 'flower_arrangement.n.01',
    'synonyms': ['flower_arrangement', 'floral_arrangement'],
    'def': 'a decorative arrangement of flowers',
    'name': 'flower_arrangement'
}, {
    'frequency': 'c',
    'id': 468,
    'synset': 'flute.n.02',
    'synonyms': ['flute_glass', 'champagne_flute'],
    'def': 'a tall narrow wineglass',
    'name': 'flute_glass'
}, {
    'frequency': 'r',
    'id': 469,
    'synset': 'foal.n.01',
    'synonyms': ['foal'],
    'def': 'a young horse',
    'name': 'foal'
}, {
    'frequency': 'c',
    'id': 470,
    'synset': 'folding_chair.n.01',
    'synonyms': ['folding_chair'],
    'def': 'a chair that can be folded flat for storage',
    'name': 'folding_chair'
}, {
    'frequency':
        'c',
    'id':
        471,
    'synset':
        'food_processor.n.01',
    'synonyms': ['food_processor'],
    'def':
        'a kitchen appliance for shredding, blending, chopping, or slicing food',
    'name':
        'food_processor'
}, {
    'frequency': 'c',
    'id': 472,
    'synset': 'football.n.02',
    'synonyms': ['football_(American)'],
    'def': 'the inflated oblong ball used in playing American football',
    'name': 'football_(American)'
}, {
    'frequency':
        'r',
    'id':
        473,
    'synset':
        'football_helmet.n.01',
    'synonyms': ['football_helmet'],
    'def':
        'a padded helmet with a face mask to protect the head of football players',
    'name':
        'football_helmet'
}, {
    'frequency': 'c',
    'id': 474,
    'synset': 'footstool.n.01',
    'synonyms': ['footstool', 'footrest'],
    'def': 'a low seat or a stool to rest the feet of a seated person',
    'name': 'footstool'
}, {
    'frequency': 'f',
    'id': 475,
    'synset': 'fork.n.01',
    'synonyms': ['fork'],
    'def': 'cutlery used for serving and eating food',
    'name': 'fork'
}, {
    'frequency':
        'r',
    'id':
        476,
    'synset':
        'forklift.n.01',
    'synonyms': ['forklift'],
    'def':
        'an industrial vehicle with a power operated fork in front that can be inserted under loads to lift and move them',
    'name':
        'forklift'
}, {
    'frequency': 'r',
    'id': 477,
    'synset': 'freight_car.n.01',
    'synonyms': ['freight_car'],
    'def': 'a railway car that carries freight',
    'name': 'freight_car'
}, {
    'frequency': 'r',
    'id': 478,
    'synset': 'french_toast.n.01',
    'synonyms': ['French_toast'],
    'def': 'bread slice dipped in egg and milk and fried',
    'name': 'French_toast'
}, {
    'frequency': 'c',
    'id': 479,
    'synset': 'freshener.n.01',
    'synonyms': ['freshener', 'air_freshener'],
    'def': 'anything that freshens',
    'name': 'freshener'
}, {
    'frequency':
        'f',
    'id':
        480,
    'synset':
        'frisbee.n.01',
    'synonyms': ['frisbee'],
    'def':
        'a light, plastic disk propelled with a flip of the wrist for recreation or competition',
    'name':
        'frisbee'
}, {
    'frequency':
        'c',
    'id':
        481,
    'synset':
        'frog.n.01',
    'synonyms': ['frog', 'toad', 'toad_frog'],
    'def':
        'a tailless stout-bodied amphibians with long hind limbs for leaping',
    'name':
        'frog'
}, {
    'frequency': 'c',
    'id': 482,
    'synset': 'fruit_juice.n.01',
    'synonyms': ['fruit_juice'],
    'def': 'drink produced by squeezing or crushing fruit',
    'name': 'fruit_juice'
}, {
    'frequency': 'r',
    'id': 483,
    'synset': 'fruit_salad.n.01',
    'synonyms': ['fruit_salad'],
    'def': 'salad composed of fruits',
    'name': 'fruit_salad'
}, {
    'frequency': 'c',
    'id': 484,
    'synset': 'frying_pan.n.01',
    'synonyms': ['frying_pan', 'frypan', 'skillet'],
    'def': 'a pan used for frying foods',
    'name': 'frying_pan'
}, {
    'frequency': 'r',
    'id': 485,
    'synset': 'fudge.n.01',
    'synonyms': ['fudge'],
    'def': 'soft creamy candy',
    'name': 'fudge'
}, {
    'frequency':
        'r',
    'id':
        486,
    'synset':
        'funnel.n.02',
    'synonyms': ['funnel'],
    'def':
        'a cone-shaped utensil used to channel a substance into a container with a small mouth',
    'name':
        'funnel'
}, {
    'frequency': 'c',
    'id': 487,
    'synset': 'futon.n.01',
    'synonyms': ['futon'],
    'def': 'a pad that is used for sleeping on the floor or on a raised frame',
    'name': 'futon'
}, {
    'frequency':
        'r',
    'id':
        488,
    'synset':
        'gag.n.02',
    'synonyms': ['gag', 'muzzle'],
    'def':
        "restraint put into a person's mouth to prevent speaking or shouting",
    'name':
        'gag'
}, {
    'frequency': 'r',
    'id': 489,
    'synset': 'garbage.n.03',
    'synonyms': ['garbage'],
    'def': 'a receptacle where waste can be discarded',
    'name': 'garbage'
}, {
    'frequency': 'c',
    'id': 490,
    'synset': 'garbage_truck.n.01',
    'synonyms': ['garbage_truck'],
    'def': 'a truck for collecting domestic refuse',
    'name': 'garbage_truck'
}, {
    'frequency': 'c',
    'id': 491,
    'synset': 'garden_hose.n.01',
    'synonyms': ['garden_hose'],
    'def': 'a hose used for watering a lawn or garden',
    'name': 'garden_hose'
}, {
    'frequency': 'c',
    'id': 492,
    'synset': 'gargle.n.01',
    'synonyms': ['gargle', 'mouthwash'],
    'def': 'a medicated solution used for gargling and rinsing the mouth',
    'name': 'gargle'
}, {
    'frequency':
        'r',
    'id':
        493,
    'synset':
        'gargoyle.n.02',
    'synonyms': ['gargoyle'],
    'def':
        'an ornament consisting of a grotesquely carved figure of a person or animal',
    'name':
        'gargoyle'
}, {
    'frequency': 'c',
    'id': 494,
    'synset': 'garlic.n.02',
    'synonyms': ['garlic', 'ail'],
    'def': 'aromatic bulb used as seasoning',
    'name': 'garlic'
}, {
    'frequency': 'r',
    'id': 495,
    'synset': 'gasmask.n.01',
    'synonyms': ['gasmask', 'respirator', 'gas_helmet'],
    'def': 'a protective face mask with a filter',
    'name': 'gasmask'
}, {
    'frequency':
        'r',
    'id':
        496,
    'synset':
        'gazelle.n.01',
    'synonyms': ['gazelle'],
    'def':
        'small swift graceful antelope of Africa and Asia having lustrous eyes',
    'name':
        'gazelle'
}, {
    'frequency':
        'c',
    'id':
        497,
    'synset':
        'gelatin.n.02',
    'synonyms': ['gelatin', 'jelly'],
    'def':
        'an edible jelly made with gelatin and used as a dessert or salad base or a coating for foods',
    'name':
        'gelatin'
}, {
    'frequency': 'r',
    'id': 498,
    'synset': 'gem.n.02',
    'synonyms': ['gemstone'],
    'def': 'a crystalline rock that can be cut and polished for jewelry',
    'name': 'gemstone'
}, {
    'frequency':
        'c',
    'id':
        499,
    'synset':
        'giant_panda.n.01',
    'synonyms': ['giant_panda', 'panda', 'panda_bear'],
    'def':
        'large black-and-white herbivorous mammal of bamboo forests of China and Tibet',
    'name':
        'giant_panda'
}, {
    'frequency': 'c',
    'id': 500,
    'synset': 'gift_wrap.n.01',
    'synonyms': ['gift_wrap'],
    'def': 'attractive wrapping paper suitable for wrapping gifts',
    'name': 'gift_wrap'
}, {
    'frequency': 'c',
    'id': 501,
    'synset': 'ginger.n.03',
    'synonyms': ['ginger', 'gingerroot'],
    'def': 'the root of the common ginger plant; used fresh as a seasoning',
    'name': 'ginger'
}, {
    'frequency':
        'f',
    'id':
        502,
    'synset':
        'giraffe.n.01',
    'synonyms': ['giraffe'],
    'def':
        'tall animal having a spotted coat and small horns and very long neck and legs',
    'name':
        'giraffe'
}, {
    'frequency':
        'c',
    'id':
        503,
    'synset':
        'girdle.n.02',
    'synonyms': ['cincture', 'sash', 'waistband', 'waistcloth'],
    'def':
        'a band of material around the waist that strengthens a skirt or trousers',
    'name':
        'cincture'
}, {
    'frequency': 'f',
    'id': 504,
    'synset': 'glass.n.02',
    'synonyms': ['glass_(drink_container)', 'drinking_glass'],
    'def': 'a container for holding liquids while drinking',
    'name': 'glass_(drink_container)'
}, {
    'frequency': 'c',
    'id': 505,
    'synset': 'globe.n.03',
    'synonyms': ['globe'],
    'def': 'a sphere on which a map (especially of the earth) is represented',
    'name': 'globe'
}, {
    'frequency': 'f',
    'id': 506,
    'synset': 'glove.n.02',
    'synonyms': ['glove'],
    'def': 'handwear covering the hand',
    'name': 'glove'
}, {
    'frequency': 'c',
    'id': 507,
    'synset': 'goat.n.01',
    'synonyms': ['goat'],
    'def': 'a common goat',
    'name': 'goat'
}, {
    'frequency': 'f',
    'id': 508,
    'synset': 'goggles.n.01',
    'synonyms': ['goggles'],
    'def': 'tight-fitting spectacles worn to protect the eyes',
    'name': 'goggles'
}, {
    'frequency':
        'r',
    'id':
        509,
    'synset':
        'goldfish.n.01',
    'synonyms': ['goldfish'],
    'def':
        'small golden or orange-red freshwater fishes used as pond or aquarium pets',
    'name':
        'goldfish'
}, {
    'frequency': 'r',
    'id': 510,
    'synset': 'golf_club.n.02',
    'synonyms': ['golf_club', 'golf-club'],
    'def': 'golf equipment used by a golfer to hit a golf ball',
    'name': 'golf_club'
}, {
    'frequency': 'c',
    'id': 511,
    'synset': 'golfcart.n.01',
    'synonyms': ['golfcart'],
    'def': 'a small motor vehicle in which golfers can ride between shots',
    'name': 'golfcart'
}, {
    'frequency':
        'r',
    'id':
        512,
    'synset':
        'gondola.n.02',
    'synonyms': ['gondola_(boat)'],
    'def':
        'long narrow flat-bottomed boat propelled by sculling; traditionally used on canals of Venice',
    'name':
        'gondola_(boat)'
}, {
    'frequency':
        'c',
    'id':
        513,
    'synset':
        'goose.n.01',
    'synonyms': ['goose'],
    'def':
        'loud, web-footed long-necked aquatic birds usually larger than ducks',
    'name':
        'goose'
}, {
    'frequency': 'r',
    'id': 514,
    'synset': 'gorilla.n.01',
    'synonyms': ['gorilla'],
    'def': 'largest ape',
    'name': 'gorilla'
}, {
    'frequency': 'r',
    'id': 515,
    'synset': 'gourd.n.02',
    'synonyms': ['gourd'],
    'def': 'any of numerous inedible fruits with hard rinds',
    'name': 'gourd'
}, {
    'frequency': 'r',
    'id': 516,
    'synset': 'gown.n.04',
    'synonyms': ['surgical_gown', 'scrubs_(surgical_clothing)'],
    'def': 'protective garment worn by surgeons during operations',
    'name': 'surgical_gown'
}, {
    'frequency':
        'f',
    'id':
        517,
    'synset':
        'grape.n.01',
    'synonyms': ['grape'],
    'def':
        'any of various juicy fruit with green or purple skins; grow in clusters',
    'name':
        'grape'
}, {
    'frequency': 'r',
    'id': 518,
    'synset': 'grasshopper.n.01',
    'synonyms': ['grasshopper'],
    'def': 'plant-eating insect with hind legs adapted for leaping',
    'name': 'grasshopper'
}, {
    'frequency':
        'c',
    'id':
        519,
    'synset':
        'grater.n.01',
    'synonyms': ['grater'],
    'def':
        'utensil with sharp perforations for shredding foods (as vegetables or cheese)',
    'name':
        'grater'
}, {
    'frequency': 'c',
    'id': 520,
    'synset': 'gravestone.n.01',
    'synonyms': ['gravestone', 'headstone', 'tombstone'],
    'def': 'a stone that is used to mark a grave',
    'name': 'gravestone'
}, {
    'frequency': 'r',
    'id': 521,
    'synset': 'gravy_boat.n.01',
    'synonyms': ['gravy_boat', 'gravy_holder'],
    'def': 'a dish (often boat-shaped) for serving gravy or sauce',
    'name': 'gravy_boat'
}, {
    'frequency': 'c',
    'id': 522,
    'synset': 'green_bean.n.02',
    'synonyms': ['green_bean'],
    'def': 'a common bean plant cultivated for its slender green edible pods',
    'name': 'green_bean'
}, {
    'frequency': 'c',
    'id': 523,
    'synset': 'green_onion.n.01',
    'synonyms': ['green_onion', 'spring_onion', 'scallion'],
    'def': 'a young onion before the bulb has enlarged',
    'name': 'green_onion'
}, {
    'frequency':
        'r',
    'id':
        524,
    'synset':
        'griddle.n.01',
    'synonyms': ['griddle'],
    'def':
        'cooking utensil consisting of a flat heated surface on which food is cooked',
    'name':
        'griddle'
}, {
    'frequency': 'r',
    'id': 525,
    'synset': 'grillroom.n.01',
    'synonyms': ['grillroom', 'grill_(restaurant)'],
    'def': 'a restaurant where food is cooked on a grill',
    'name': 'grillroom'
}, {
    'frequency': 'r',
    'id': 526,
    'synset': 'grinder.n.04',
    'synonyms': ['grinder_(tool)'],
    'def': 'a machine tool that polishes metal',
    'name': 'grinder_(tool)'
}, {
    'frequency': 'r',
    'id': 527,
    'synset': 'grits.n.01',
    'synonyms': ['grits', 'hominy_grits'],
    'def': 'coarsely ground corn boiled as a breakfast dish',
    'name': 'grits'
}, {
    'frequency':
        'c',
    'id':
        528,
    'synset':
        'grizzly.n.01',
    'synonyms': ['grizzly', 'grizzly_bear'],
    'def':
        'powerful brownish-yellow bear of the uplands of western North America',
    'name':
        'grizzly'
}, {
    'frequency': 'c',
    'id': 529,
    'synset': 'grocery_bag.n.01',
    'synonyms': ['grocery_bag'],
    'def': "a sack for holding customer's groceries",
    'name': 'grocery_bag'
}, {
    'frequency':
        'r',
    'id':
        530,
    'synset':
        'guacamole.n.01',
    'synonyms': ['guacamole'],
    'def':
        'a dip made of mashed avocado mixed with chopped onions and other seasonings',
    'name':
        'guacamole'
}, {
    'frequency':
        'f',
    'id':
        531,
    'synset':
        'guitar.n.01',
    'synonyms': ['guitar'],
    'def':
        'a stringed instrument usually having six strings; played by strumming or plucking',
    'name':
        'guitar'
}, {
    'frequency': 'c',
    'id': 532,
    'synset': 'gull.n.02',
    'synonyms': ['gull', 'seagull'],
    'def': 'mostly white aquatic bird having long pointed wings and short legs',
    'name': 'gull'
}, {
    'frequency':
        'c',
    'id':
        533,
    'synset':
        'gun.n.01',
    'synonyms': ['gun'],
    'def':
        'a weapon that discharges a bullet at high velocity from a metal tube',
    'name':
        'gun'
}, {
    'frequency': 'r',
    'id': 534,
    'synset': 'hair_spray.n.01',
    'synonyms': ['hair_spray'],
    'def': 'substance sprayed on the hair to hold it in place',
    'name': 'hair_spray'
}, {
    'frequency': 'c',
    'id': 535,
    'synset': 'hairbrush.n.01',
    'synonyms': ['hairbrush'],
    'def': "a brush used to groom a person's hair",
    'name': 'hairbrush'
}, {
    'frequency': 'c',
    'id': 536,
    'synset': 'hairnet.n.01',
    'synonyms': ['hairnet'],
    'def': 'a small net that someone wears over their hair to keep it in place',
    'name': 'hairnet'
}, {
    'frequency': 'c',
    'id': 537,
    'synset': 'hairpin.n.01',
    'synonyms': ['hairpin'],
    'def': "a double pronged pin used to hold women's hair in place",
    'name': 'hairpin'
}, {
    'frequency': 'f',
    'id': 538,
    'synset': 'ham.n.01',
    'synonyms': ['ham', 'jambon', 'gammon'],
    'def': 'meat cut from the thigh of a hog (usually smoked)',
    'name': 'ham'
}, {
    'frequency': 'c',
    'id': 539,
    'synset': 'hamburger.n.01',
    'synonyms': ['hamburger', 'beefburger', 'burger'],
    'def': 'a sandwich consisting of a patty of minced beef served on a bun',
    'name': 'hamburger'
}, {
    'frequency':
        'c',
    'id':
        540,
    'synset':
        'hammer.n.02',
    'synonyms': ['hammer'],
    'def':
        'a hand tool with a heavy head and a handle; used to deliver an impulsive force by striking',
    'name':
        'hammer'
}, {
    'frequency':
        'r',
    'id':
        541,
    'synset':
        'hammock.n.02',
    'synonyms': ['hammock'],
    'def':
        'a hanging bed of canvas or rope netting (usually suspended between two trees)',
    'name':
        'hammock'
}, {
    'frequency': 'r',
    'id': 542,
    'synset': 'hamper.n.02',
    'synonyms': ['hamper'],
    'def': 'a basket usually with a cover',
    'name': 'hamper'
}, {
    'frequency': 'r',
    'id': 543,
    'synset': 'hamster.n.01',
    'synonyms': ['hamster'],
    'def': 'short-tailed burrowing rodent with large cheek pouches',
    'name': 'hamster'
}, {
    'frequency': 'c',
    'id': 544,
    'synset': 'hand_blower.n.01',
    'synonyms': ['hair_dryer'],
    'def': 'a hand-held electric blower that can blow warm air onto the hair',
    'name': 'hair_dryer'
}, {
    'frequency': 'r',
    'id': 545,
    'synset': 'hand_glass.n.01',
    'synonyms': ['hand_glass', 'hand_mirror'],
    'def': 'a mirror intended to be held in the hand',
    'name': 'hand_glass'
}, {
    'frequency': 'f',
    'id': 546,
    'synset': 'hand_towel.n.01',
    'synonyms': ['hand_towel', 'face_towel'],
    'def': 'a small towel used to dry the hands or face',
    'name': 'hand_towel'
}, {
    'frequency': 'c',
    'id': 547,
    'synset': 'handcart.n.01',
    'synonyms': ['handcart', 'pushcart', 'hand_truck'],
    'def': 'wheeled vehicle that can be pushed by a person',
    'name': 'handcart'
}, {
    'frequency':
        'r',
    'id':
        548,
    'synset':
        'handcuff.n.01',
    'synonyms': ['handcuff'],
    'def':
        'shackle that consists of a metal loop that can be locked around the wrist',
    'name':
        'handcuff'
}, {
    'frequency':
        'c',
    'id':
        549,
    'synset':
        'handkerchief.n.01',
    'synonyms': ['handkerchief'],
    'def':
        'a square piece of cloth used for wiping the eyes or nose or as a costume accessory',
    'name':
        'handkerchief'
}, {
    'frequency':
        'f',
    'id':
        550,
    'synset':
        'handle.n.01',
    'synonyms': ['handle', 'grip', 'handgrip'],
    'def':
        'the appendage to an object that is designed to be held in order to use or move it',
    'name':
        'handle'
}, {
    'frequency': 'r',
    'id': 551,
    'synset': 'handsaw.n.01',
    'synonyms': ['handsaw', "carpenter's_saw"],
    'def': 'a saw used with one hand for cutting wood',
    'name': 'handsaw'
}, {
    'frequency': 'r',
    'id': 552,
    'synset': 'hardback.n.01',
    'synonyms': ['hardback_book', 'hardcover_book'],
    'def': 'a book with cardboard or cloth or leather covers',
    'name': 'hardback_book'
}, {
    'frequency':
        'r',
    'id':
        553,
    'synset':
        'harmonium.n.01',
    'synonyms': [
        'harmonium', 'organ_(musical_instrument)',
        'reed_organ_(musical_instrument)'
    ],
    'def':
        'a free-reed instrument in which air is forced through the reeds by bellows',
    'name':
        'harmonium'
}, {
    'frequency':
        'f',
    'id':
        554,
    'synset':
        'hat.n.01',
    'synonyms': ['hat'],
    'def':
        'headwear that protects the head from bad weather, sun, or worn for fashion',
    'name':
        'hat'
}, {
    'frequency': 'r',
    'id': 555,
    'synset': 'hatbox.n.01',
    'synonyms': ['hatbox'],
    'def': 'a round piece of luggage for carrying hats',
    'name': 'hatbox'
}, {
    'frequency': 'r',
    'id': 556,
    'synset': 'hatch.n.03',
    'synonyms': ['hatch'],
    'def': 'a movable barrier covering a hatchway',
    'name': 'hatch'
}, {
    'frequency': 'c',
    'id': 557,
    'synset': 'head_covering.n.01',
    'synonyms': ['veil'],
    'def': 'a garment that covers the head and face',
    'name': 'veil'
}, {
    'frequency': 'f',
    'id': 558,
    'synset': 'headband.n.01',
    'synonyms': ['headband'],
    'def': 'a band worn around or over the head',
    'name': 'headband'
}, {
    'frequency': 'f',
    'id': 559,
    'synset': 'headboard.n.01',
    'synonyms': ['headboard'],
    'def': 'a vertical board or panel forming the head of a bedstead',
    'name': 'headboard'
}, {
    'frequency':
        'f',
    'id':
        560,
    'synset':
        'headlight.n.01',
    'synonyms': ['headlight', 'headlamp'],
    'def':
        'a powerful light with reflector; attached to the front of an automobile or locomotive',
    'name':
        'headlight'
}, {
    'frequency': 'c',
    'id': 561,
    'synset': 'headscarf.n.01',
    'synonyms': ['headscarf'],
    'def': 'a kerchief worn over the head and tied under the chin',
    'name': 'headscarf'
}, {
    'frequency': 'r',
    'id': 562,
    'synset': 'headset.n.01',
    'synonyms': ['headset'],
    'def': 'receiver consisting of a pair of headphones',
    'name': 'headset'
}, {
    'frequency':
        'c',
    'id':
        563,
    'synset':
        'headstall.n.01',
    'synonyms': ['headstall_(for_horses)', 'headpiece_(for_horses)'],
    'def':
        "the band that is the part of a bridle that fits around a horse's head",
    'name':
        'headstall_(for_horses)'
}, {
    'frequency':
        'r',
    'id':
        564,
    'synset':
        'hearing_aid.n.02',
    'synonyms': ['hearing_aid'],
    'def':
        'an acoustic device used to direct sound to the ear of a hearing-impaired person',
    'name':
        'hearing_aid'
}, {
    'frequency': 'c',
    'id': 565,
    'synset': 'heart.n.02',
    'synonyms': ['heart'],
    'def': 'a muscular organ; its contractions move the blood through the body',
    'name': 'heart'
}, {
    'frequency': 'c',
    'id': 566,
    'synset': 'heater.n.01',
    'synonyms': ['heater', 'warmer'],
    'def': 'device that heats water or supplies warmth to a room',
    'name': 'heater'
}, {
    'frequency':
        'c',
    'id':
        567,
    'synset':
        'helicopter.n.01',
    'synonyms': ['helicopter'],
    'def':
        'an aircraft without wings that obtains its lift from the rotation of overhead blades',
    'name':
        'helicopter'
}, {
    'frequency': 'f',
    'id': 568,
    'synset': 'helmet.n.02',
    'synonyms': ['helmet'],
    'def': 'a protective headgear made of hard material to resist blows',
    'name': 'helmet'
}, {
    'frequency':
        'r',
    'id':
        569,
    'synset':
        'heron.n.02',
    'synonyms': ['heron'],
    'def':
        'grey or white wading bird with long neck and long legs and (usually) long bill',
    'name':
        'heron'
}, {
    'frequency': 'c',
    'id': 570,
    'synset': 'highchair.n.01',
    'synonyms': ['highchair', 'feeding_chair'],
    'def': 'a chair for feeding a very young child',
    'name': 'highchair'
}, {
    'frequency':
        'f',
    'id':
        571,
    'synset':
        'hinge.n.01',
    'synonyms': ['hinge'],
    'def':
        'a joint that holds two parts together so that one can swing relative to the other',
    'name':
        'hinge'
}, {
    'frequency':
        'r',
    'id':
        572,
    'synset':
        'hippopotamus.n.01',
    'synonyms': ['hippopotamus'],
    'def':
        'massive thick-skinned animal living in or around rivers of tropical Africa',
    'name':
        'hippopotamus'
}, {
    'frequency':
        'r',
    'id':
        573,
    'synset':
        'hockey_stick.n.01',
    'synonyms': ['hockey_stick'],
    'def':
        'sports implement consisting of a stick used by hockey players to move the puck',
    'name':
        'hockey_stick'
}, {
    'frequency': 'c',
    'id': 574,
    'synset': 'hog.n.03',
    'synonyms': ['hog', 'pig'],
    'def': 'domestic swine',
    'name': 'hog'
}, {
    'frequency':
        'f',
    'id':
        575,
    'synset':
        'home_plate.n.01',
    'synonyms': ['home_plate_(baseball)', 'home_base_(baseball)'],
    'def':
        '(baseball) a rubber slab where the batter stands; it must be touched by a base runner in order to score',
    'name':
        'home_plate_(baseball)'
}, {
    'frequency': 'c',
    'id': 576,
    'synset': 'honey.n.01',
    'synonyms': ['honey'],
    'def': 'a sweet yellow liquid produced by bees',
    'name': 'honey'
}, {
    'frequency': 'f',
    'id': 577,
    'synset': 'hood.n.06',
    'synonyms': ['fume_hood', 'exhaust_hood'],
    'def': 'metal covering leading to a vent that exhausts smoke or fumes',
    'name': 'fume_hood'
}, {
    'frequency': 'f',
    'id': 578,
    'synset': 'hook.n.05',
    'synonyms': ['hook'],
    'def': 'a curved or bent implement for suspending or pulling something',
    'name': 'hook'
}, {
    'frequency': 'f',
    'id': 579,
    'synset': 'horse.n.01',
    'synonyms': ['horse'],
    'def': 'a common horse',
    'name': 'horse'
}, {
    'frequency': 'f',
    'id': 580,
    'synset': 'hose.n.03',
    'synonyms': ['hose', 'hosepipe'],
    'def': 'a flexible pipe for conveying a liquid or gas',
    'name': 'hose'
}, {
    'frequency':
        'r',
    'id':
        581,
    'synset':
        'hot-air_balloon.n.01',
    'synonyms': ['hot-air_balloon'],
    'def':
        'balloon for travel through the air in a basket suspended below a large bag of heated air',
    'name':
        'hot-air_balloon'
}, {
    'frequency':
        'r',
    'id':
        582,
    'synset':
        'hot_plate.n.01',
    'synonyms': ['hotplate'],
    'def':
        'a portable electric appliance for heating or cooking or keeping food warm',
    'name':
        'hotplate'
}, {
    'frequency': 'c',
    'id': 583,
    'synset': 'hot_sauce.n.01',
    'synonyms': ['hot_sauce'],
    'def': 'a pungent peppery sauce',
    'name': 'hot_sauce'
}, {
    'frequency': 'r',
    'id': 584,
    'synset': 'hourglass.n.01',
    'synonyms': ['hourglass'],
    'def': 'a sandglass timer that runs for sixty minutes',
    'name': 'hourglass'
}, {
    'frequency': 'r',
    'id': 585,
    'synset': 'houseboat.n.01',
    'synonyms': ['houseboat'],
    'def': 'a barge that is designed and equipped for use as a dwelling',
    'name': 'houseboat'
}, {
    'frequency':
        'r',
    'id':
        586,
    'synset':
        'hummingbird.n.01',
    'synonyms': ['hummingbird'],
    'def':
        'tiny American bird having brilliant iridescent plumage and long slender bills',
    'name':
        'hummingbird'
}, {
    'frequency': 'r',
    'id': 587,
    'synset': 'hummus.n.01',
    'synonyms': ['hummus', 'humus', 'hommos', 'hoummos', 'humous'],
    'def': 'a thick spread made from mashed chickpeas',
    'name': 'hummus'
}, {
    'frequency': 'c',
    'id': 588,
    'synset': 'ice_bear.n.01',
    'synonyms': ['polar_bear'],
    'def': 'white bear of Arctic regions',
    'name': 'polar_bear'
}, {
    'frequency': 'c',
    'id': 589,
    'synset': 'ice_cream.n.01',
    'synonyms': ['icecream'],
    'def': 'frozen dessert containing cream and sugar and flavoring',
    'name': 'icecream'
}, {
    'frequency': 'r',
    'id': 590,
    'synset': 'ice_lolly.n.01',
    'synonyms': ['popsicle'],
    'def': 'ice cream or water ice on a small wooden stick',
    'name': 'popsicle'
}, {
    'frequency':
        'c',
    'id':
        591,
    'synset':
        'ice_maker.n.01',
    'synonyms': ['ice_maker'],
    'def':
        'an appliance included in some electric refrigerators for making ice cubes',
    'name':
        'ice_maker'
}, {
    'frequency':
        'r',
    'id':
        592,
    'synset':
        'ice_pack.n.01',
    'synonyms': ['ice_pack', 'ice_bag'],
    'def':
        'a waterproof bag filled with ice: applied to the body (especially the head) to cool or reduce swelling',
    'name':
        'ice_pack'
}, {
    'frequency': 'r',
    'id': 593,
    'synset': 'ice_skate.n.01',
    'synonyms': ['ice_skate'],
    'def': 'skate consisting of a boot with a steel blade fitted to the sole',
    'name': 'ice_skate'
}, {
    'frequency': 'r',
    'id': 594,
    'synset': 'ice_tea.n.01',
    'synonyms': ['ice_tea', 'iced_tea'],
    'def': 'strong tea served over ice',
    'name': 'ice_tea'
}, {
    'frequency': 'c',
    'id': 595,
    'synset': 'igniter.n.01',
    'synonyms': ['igniter', 'ignitor', 'lighter'],
    'def': 'a substance or device used to start a fire',
    'name': 'igniter'
}, {
    'frequency': 'r',
    'id': 596,
    'synset': 'incense.n.01',
    'synonyms': ['incense'],
    'def': 'a substance that produces a fragrant odor when burned',
    'name': 'incense'
}, {
    'frequency':
        'r',
    'id':
        597,
    'synset':
        'inhaler.n.01',
    'synonyms': ['inhaler', 'inhalator'],
    'def':
        'a dispenser that produces a chemical vapor to be inhaled through mouth or nose',
    'name':
        'inhaler'
}, {
    'frequency': 'c',
    'id': 598,
    'synset': 'ipod.n.01',
    'synonyms': ['iPod'],
    'def': 'a pocket-sized device used to play music files',
    'name': 'iPod'
}, {
    'frequency':
        'c',
    'id':
        599,
    'synset':
        'iron.n.04',
    'synonyms': ['iron_(for_clothing)', 'smoothing_iron_(for_clothing)'],
    'def':
        'home appliance consisting of a flat metal base that is heated and used to smooth cloth',
    'name':
        'iron_(for_clothing)'
}, {
    'frequency':
        'r',
    'id':
        600,
    'synset':
        'ironing_board.n.01',
    'synonyms': ['ironing_board'],
    'def':
        'narrow padded board on collapsible supports; used for ironing clothes',
    'name':
        'ironing_board'
}, {
    'frequency': 'f',
    'id': 601,
    'synset': 'jacket.n.01',
    'synonyms': ['jacket'],
    'def': 'a waist-length coat',
    'name': 'jacket'
}, {
    'frequency': 'r',
    'id': 602,
    'synset': 'jam.n.01',
    'synonyms': ['jam'],
    'def': 'preserve of crushed fruit',
    'name': 'jam'
}, {
    'frequency':
        'f',
    'id':
        603,
    'synset':
        'jean.n.01',
    'synonyms': ['jean', 'blue_jean', 'denim'],
    'def':
        '(usually plural) close-fitting trousers of heavy denim for manual work or casual wear',
    'name':
        'jean'
}, {
    'frequency': 'c',
    'id': 604,
    'synset': 'jeep.n.01',
    'synonyms': ['jeep', 'landrover'],
    'def': 'a car suitable for traveling over rough terrain',
    'name': 'jeep'
}, {
    'frequency': 'r',
    'id': 605,
    'synset': 'jelly_bean.n.01',
    'synonyms': ['jelly_bean', 'jelly_egg'],
    'def': 'sugar-glazed jellied candy',
    'name': 'jelly_bean'
}, {
    'frequency': 'f',
    'id': 606,
    'synset': 'jersey.n.03',
    'synonyms': ['jersey', 'T-shirt', 'tee_shirt'],
    'def': 'a close-fitting pullover shirt',
    'name': 'jersey'
}, {
    'frequency': 'c',
    'id': 607,
    'synset': 'jet.n.01',
    'synonyms': ['jet_plane', 'jet-propelled_plane'],
    'def': 'an airplane powered by one or more jet engines',
    'name': 'jet_plane'
}, {
    'frequency':
        'c',
    'id':
        608,
    'synset':
        'jewelry.n.01',
    'synonyms': ['jewelry', 'jewellery'],
    'def':
        'an adornment (as a bracelet or ring or necklace) made of precious metals and set with gems (or imitation gems)',
    'name':
        'jewelry'
}, {
    'frequency':
        'r',
    'id':
        609,
    'synset':
        'joystick.n.02',
    'synonyms': ['joystick'],
    'def':
        'a control device for computers consisting of a vertical handle that can move freely in two directions',
    'name':
        'joystick'
}, {
    'frequency': 'r',
    'id': 610,
    'synset': 'jump_suit.n.01',
    'synonyms': ['jumpsuit'],
    'def': "one-piece garment fashioned after a parachutist's uniform",
    'name': 'jumpsuit'
}, {
    'frequency':
        'c',
    'id':
        611,
    'synset':
        'kayak.n.01',
    'synonyms': ['kayak'],
    'def':
        'a small canoe consisting of a light frame made watertight with animal skins',
    'name':
        'kayak'
}, {
    'frequency': 'r',
    'id': 612,
    'synset': 'keg.n.02',
    'synonyms': ['keg'],
    'def': 'small cask or barrel',
    'name': 'keg'
}, {
    'frequency': 'r',
    'id': 613,
    'synset': 'kennel.n.01',
    'synonyms': ['kennel', 'doghouse'],
    'def': 'outbuilding that serves as a shelter for a dog',
    'name': 'kennel'
}, {
    'frequency': 'c',
    'id': 614,
    'synset': 'kettle.n.01',
    'synonyms': ['kettle', 'boiler'],
    'def': 'a metal pot for stewing or boiling; usually has a lid',
    'name': 'kettle'
}, {
    'frequency': 'f',
    'id': 615,
    'synset': 'key.n.01',
    'synonyms': ['key'],
    'def': 'metal instrument used to unlock a lock',
    'name': 'key'
}, {
    'frequency': 'r',
    'id': 616,
    'synset': 'keycard.n.01',
    'synonyms': ['keycard'],
    'def': 'a plastic card used to gain access typically to a door',
    'name': 'keycard'
}, {
    'frequency':
        'r',
    'id':
        617,
    'synset':
        'kilt.n.01',
    'synonyms': ['kilt'],
    'def':
        'a knee-length pleated tartan skirt worn by men as part of the traditional dress in the Highlands of northern Scotland',
    'name':
        'kilt'
}, {
    'frequency': 'c',
    'id': 618,
    'synset': 'kimono.n.01',
    'synonyms': ['kimono'],
    'def': 'a loose robe; imitated from robes originally worn by Japanese',
    'name': 'kimono'
}, {
    'frequency': 'f',
    'id': 619,
    'synset': 'kitchen_sink.n.01',
    'synonyms': ['kitchen_sink'],
    'def': 'a sink in a kitchen',
    'name': 'kitchen_sink'
}, {
    'frequency': 'c',
    'id': 620,
    'synset': 'kitchen_table.n.01',
    'synonyms': ['kitchen_table'],
    'def': 'a table in the kitchen',
    'name': 'kitchen_table'
}, {
    'frequency':
        'f',
    'id':
        621,
    'synset':
        'kite.n.03',
    'synonyms': ['kite'],
    'def':
        'plaything consisting of a light frame covered with tissue paper; flown in wind at end of a string',
    'name':
        'kite'
}, {
    'frequency': 'c',
    'id': 622,
    'synset': 'kitten.n.01',
    'synonyms': ['kitten', 'kitty'],
    'def': 'young domestic cat',
    'name': 'kitten'
}, {
    'frequency': 'c',
    'id': 623,
    'synset': 'kiwi.n.03',
    'synonyms': ['kiwi_fruit'],
    'def': 'fuzzy brown egg-shaped fruit with slightly tart green flesh',
    'name': 'kiwi_fruit'
}, {
    'frequency':
        'f',
    'id':
        624,
    'synset':
        'knee_pad.n.01',
    'synonyms': ['knee_pad'],
    'def':
        'protective garment consisting of a pad worn by football or baseball or hockey players',
    'name':
        'knee_pad'
}, {
    'frequency': 'f',
    'id': 625,
    'synset': 'knife.n.01',
    'synonyms': ['knife'],
    'def': 'tool with a blade and point used as a cutting instrument',
    'name': 'knife'
}, {
    'frequency': 'r',
    'id': 626,
    'synset': 'knight.n.02',
    'synonyms': ['knight_(chess_piece)', 'horse_(chess_piece)'],
    'def': 'a chess game piece shaped to resemble the head of a horse',
    'name': 'knight_(chess_piece)'
}, {
    'frequency':
        'r',
    'id':
        627,
    'synset':
        'knitting_needle.n.01',
    'synonyms': ['knitting_needle'],
    'def':
        'needle consisting of a slender rod with pointed ends; usually used in pairs',
    'name':
        'knitting_needle'
}, {
    'frequency': 'f',
    'id': 628,
    'synset': 'knob.n.02',
    'synonyms': ['knob'],
    'def': 'a round handle often found on a door',
    'name': 'knob'
}, {
    'frequency':
        'r',
    'id':
        629,
    'synset':
        'knocker.n.05',
    'synonyms': ['knocker_(on_a_door)', 'doorknocker'],
    'def':
        'a device (usually metal and ornamental) attached by a hinge to a door',
    'name':
        'knocker_(on_a_door)'
}, {
    'frequency':
        'r',
    'id':
        630,
    'synset':
        'koala.n.01',
    'synonyms': ['koala', 'koala_bear'],
    'def':
        'sluggish tailless Australian marsupial with grey furry ears and coat',
    'name':
        'koala'
}, {
    'frequency':
        'r',
    'id':
        631,
    'synset':
        'lab_coat.n.01',
    'synonyms': ['lab_coat', 'laboratory_coat'],
    'def':
        'a light coat worn to protect clothing from substances used while working in a laboratory',
    'name':
        'lab_coat'
}, {
    'frequency': 'f',
    'id': 632,
    'synset': 'ladder.n.01',
    'synonyms': ['ladder'],
    'def': 'steps consisting of two parallel members connected by rungs',
    'name': 'ladder'
}, {
    'frequency':
        'c',
    'id':
        633,
    'synset':
        'ladle.n.01',
    'synonyms': ['ladle'],
    'def':
        'a spoon-shaped vessel with a long handle frequently used to transfer liquids',
    'name':
        'ladle'
}, {
    'frequency':
        'r',
    'id':
        634,
    'synset':
        'ladybug.n.01',
    'synonyms': ['ladybug', 'ladybeetle', 'ladybird_beetle'],
    'def':
        'small round bright-colored and spotted beetle, typically red and black',
    'name':
        'ladybug'
}, {
    'frequency': 'c',
    'id': 635,
    'synset': 'lamb.n.01',
    'synonyms': ['lamb_(animal)'],
    'def': 'young sheep',
    'name': 'lamb_(animal)'
}, {
    'frequency': 'r',
    'id': 636,
    'synset': 'lamb_chop.n.01',
    'synonyms': ['lamb-chop', 'lambchop'],
    'def': 'chop cut from a lamb',
    'name': 'lamb-chop'
}, {
    'frequency': 'f',
    'id': 637,
    'synset': 'lamp.n.02',
    'synonyms': ['lamp'],
    'def': 'a piece of furniture holding one or more electric light bulbs',
    'name': 'lamp'
}, {
    'frequency': 'f',
    'id': 638,
    'synset': 'lamppost.n.01',
    'synonyms': ['lamppost'],
    'def': 'a metal post supporting an outdoor lamp (such as a streetlight)',
    'name': 'lamppost'
}, {
    'frequency':
        'f',
    'id':
        639,
    'synset':
        'lampshade.n.01',
    'synonyms': ['lampshade'],
    'def':
        'a protective ornamental shade used to screen a light bulb from direct view',
    'name':
        'lampshade'
}, {
    'frequency': 'c',
    'id': 640,
    'synset': 'lantern.n.01',
    'synonyms': ['lantern'],
    'def': 'light in a transparent protective case',
    'name': 'lantern'
}, {
    'frequency': 'f',
    'id': 641,
    'synset': 'lanyard.n.02',
    'synonyms': ['lanyard', 'laniard'],
    'def': 'a cord worn around the neck to hold a knife or whistle, etc.',
    'name': 'lanyard'
}, {
    'frequency': 'f',
    'id': 642,
    'synset': 'laptop.n.01',
    'synonyms': ['laptop_computer', 'notebook_computer'],
    'def': 'a portable computer small enough to use in your lap',
    'name': 'laptop_computer'
}, {
    'frequency':
        'r',
    'id':
        643,
    'synset':
        'lasagna.n.01',
    'synonyms': ['lasagna', 'lasagne'],
    'def':
        'baked dish of layers of lasagna pasta with sauce and cheese and meat or vegetables',
    'name':
        'lasagna'
}, {
    'frequency':
        'c',
    'id':
        644,
    'synset':
        'latch.n.02',
    'synonyms': ['latch'],
    'def':
        'a bar that can be lowered or slid into a groove to fasten a door or gate',
    'name':
        'latch'
}, {
    'frequency': 'r',
    'id': 645,
    'synset': 'lawn_mower.n.01',
    'synonyms': ['lawn_mower'],
    'def': 'garden tool for mowing grass on lawns',
    'name': 'lawn_mower'
}, {
    'frequency':
        'r',
    'id':
        646,
    'synset':
        'leather.n.01',
    'synonyms': ['leather'],
    'def':
        'an animal skin made smooth and flexible by removing the hair and then tanning',
    'name':
        'leather'
}, {
    'frequency':
        'c',
    'id':
        647,
    'synset':
        'legging.n.01',
    'synonyms': ['legging_(clothing)', 'leging_(clothing)', 'leg_covering'],
    'def':
        'a garment covering the leg (usually extending from the knee to the ankle)',
    'name':
        'legging_(clothing)'
}, {
    'frequency': 'c',
    'id': 648,
    'synset': 'lego.n.01',
    'synonyms': ['Lego', 'Lego_set'],
    'def': "a child's plastic construction set for making models from blocks",
    'name': 'Lego'
}, {
    'frequency': 'f',
    'id': 649,
    'synset': 'lemon.n.01',
    'synonyms': ['lemon'],
    'def': 'yellow oval fruit with juicy acidic flesh',
    'name': 'lemon'
}, {
    'frequency': 'r',
    'id': 650,
    'synset': 'lemonade.n.01',
    'synonyms': ['lemonade'],
    'def': 'sweetened beverage of diluted lemon juice',
    'name': 'lemonade'
}, {
    'frequency': 'f',
    'id': 651,
    'synset': 'lettuce.n.02',
    'synonyms': ['lettuce'],
    'def': 'leafy plant commonly eaten in salad or on sandwiches',
    'name': 'lettuce'
}, {
    'frequency':
        'f',
    'id':
        652,
    'synset':
        'license_plate.n.01',
    'synonyms': ['license_plate', 'numberplate'],
    'def':
        "a plate mounted on the front and back of car and bearing the car's registration number",
    'name':
        'license_plate'
}, {
    'frequency':
        'f',
    'id':
        653,
    'synset':
        'life_buoy.n.01',
    'synonyms': ['life_buoy', 'lifesaver', 'life_belt', 'life_ring'],
    'def':
        'a ring-shaped life preserver used to prevent drowning (NOT a life-jacket or vest)',
    'name':
        'life_buoy'
}, {
    'frequency':
        'f',
    'id':
        654,
    'synset':
        'life_jacket.n.01',
    'synonyms': ['life_jacket', 'life_vest'],
    'def':
        'life preserver consisting of a sleeveless jacket of buoyant or inflatable design',
    'name':
        'life_jacket'
}, {
    'frequency':
        'f',
    'id':
        655,
    'synset':
        'light_bulb.n.01',
    'synonyms': ['lightbulb'],
    'def':
        'glass bulb or tube shaped electric device that emits light (DO NOT MARK LAMPS AS A WHOLE)',
    'name':
        'lightbulb'
}, {
    'frequency':
        'r',
    'id':
        656,
    'synset':
        'lightning_rod.n.02',
    'synonyms': ['lightning_rod', 'lightning_conductor'],
    'def':
        'a metallic conductor that is attached to a high point and leads to the ground',
    'name':
        'lightning_rod'
}, {
    'frequency': 'c',
    'id': 657,
    'synset': 'lime.n.06',
    'synonyms': ['lime'],
    'def': 'the green acidic fruit of any of various lime trees',
    'name': 'lime'
}, {
    'frequency': 'r',
    'id': 658,
    'synset': 'limousine.n.01',
    'synonyms': ['limousine'],
    'def': 'long luxurious car; usually driven by a chauffeur',
    'name': 'limousine'
}, {
    'frequency': 'r',
    'id': 659,
    'synset': 'linen.n.02',
    'synonyms': ['linen_paper'],
    'def': 'a high-quality paper made of linen fibers or with a linen finish',
    'name': 'linen_paper'
}, {
    'frequency': 'c',
    'id': 660,
    'synset': 'lion.n.01',
    'synonyms': ['lion'],
    'def': 'large gregarious predatory cat of Africa and India',
    'name': 'lion'
}, {
    'frequency': 'c',
    'id': 661,
    'synset': 'lip_balm.n.01',
    'synonyms': ['lip_balm'],
    'def': 'a balm applied to the lips',
    'name': 'lip_balm'
}, {
    'frequency': 'c',
    'id': 662,
    'synset': 'lipstick.n.01',
    'synonyms': ['lipstick', 'lip_rouge'],
    'def': 'makeup that is used to color the lips',
    'name': 'lipstick'
}, {
    'frequency': 'r',
    'id': 663,
    'synset': 'liquor.n.01',
    'synonyms': ['liquor', 'spirits', 'hard_liquor', 'liqueur', 'cordial'],
    'def': 'an alcoholic beverage that is distilled rather than fermented',
    'name': 'liquor'
}, {
    'frequency': 'r',
    'id': 664,
    'synset': 'lizard.n.01',
    'synonyms': ['lizard'],
    'def': 'a reptile with usually two pairs of legs and a tapering tail',
    'name': 'lizard'
}, {
    'frequency': 'r',
    'id': 665,
    'synset': 'loafer.n.02',
    'synonyms': ['Loafer_(type_of_shoe)'],
    'def': 'a low leather step-in shoe',
    'name': 'Loafer_(type_of_shoe)'
}, {
    'frequency': 'f',
    'id': 666,
    'synset': 'log.n.01',
    'synonyms': ['log'],
    'def': 'a segment of the trunk of a tree when stripped of branches',
    'name': 'log'
}, {
    'frequency': 'c',
    'id': 667,
    'synset': 'lollipop.n.02',
    'synonyms': ['lollipop'],
    'def': 'hard candy on a stick',
    'name': 'lollipop'
}, {
    'frequency': 'c',
    'id': 668,
    'synset': 'lotion.n.01',
    'synonyms': ['lotion'],
    'def': 'any of various cosmetic preparations that are applied to the skin',
    'name': 'lotion'
}, {
    'frequency':
        'f',
    'id':
        669,
    'synset':
        'loudspeaker.n.01',
    'synonyms': ['speaker_(stero_equipment)'],
    'def':
        'electronic device that produces sound often as part of a stereo system',
    'name':
        'speaker_(stero_equipment)'
}, {
    'frequency': 'c',
    'id': 670,
    'synset': 'love_seat.n.01',
    'synonyms': ['loveseat'],
    'def': 'small sofa that seats two people',
    'name': 'loveseat'
}, {
    'frequency': 'r',
    'id': 671,
    'synset': 'machine_gun.n.01',
    'synonyms': ['machine_gun'],
    'def': 'a rapidly firing automatic gun',
    'name': 'machine_gun'
}, {
    'frequency': 'f',
    'id': 672,
    'synset': 'magazine.n.02',
    'synonyms': ['magazine'],
    'def': 'a paperback periodic publication',
    'name': 'magazine'
}, {
    'frequency': 'f',
    'id': 673,
    'synset': 'magnet.n.01',
    'synonyms': ['magnet'],
    'def': 'a device that attracts iron and produces a magnetic field',
    'name': 'magnet'
}, {
    'frequency': 'r',
    'id': 674,
    'synset': 'mail_slot.n.01',
    'synonyms': ['mail_slot'],
    'def': 'a slot (usually in a door) through which mail can be delivered',
    'name': 'mail_slot'
}, {
    'frequency': 'c',
    'id': 675,
    'synset': 'mailbox.n.01',
    'synonyms': ['mailbox_(at_home)', 'letter_box_(at_home)'],
    'def': 'a private box for delivery of mail',
    'name': 'mailbox_(at_home)'
}, {
    'frequency':
        'r',
    'id':
        676,
    'synset':
        'mallet.n.01',
    'synonyms': ['mallet'],
    'def':
        'a sports implement with a long handle and a hammer-like head used to hit a ball',
    'name':
        'mallet'
}, {
    'frequency':
        'r',
    'id':
        677,
    'synset':
        'mammoth.n.01',
    'synonyms': ['mammoth'],
    'def':
        'any of numerous extinct elephants widely distributed in the Pleistocene',
    'name':
        'mammoth'
}, {
    'frequency': 'c',
    'id': 678,
    'synset': 'mandarin.n.05',
    'synonyms': ['mandarin_orange'],
    'def': 'a somewhat flat reddish-orange loose skinned citrus of China',
    'name': 'mandarin_orange'
}, {
    'frequency':
        'c',
    'id':
        679,
    'synset':
        'manger.n.01',
    'synonyms': ['manger', 'trough'],
    'def':
        'a container (usually in a barn or stable) from which cattle or horses feed',
    'name':
        'manger'
}, {
    'frequency':
        'f',
    'id':
        680,
    'synset':
        'manhole.n.01',
    'synonyms': ['manhole'],
    'def':
        'a hole (usually with a flush cover) through which a person can gain access to an underground structure',
    'name':
        'manhole'
}, {
    'frequency':
        'c',
    'id':
        681,
    'synset':
        'map.n.01',
    'synonyms': ['map'],
    'def':
        "a diagrammatic representation of the earth's surface (or part of it)",
    'name':
        'map'
}, {
    'frequency': 'c',
    'id': 682,
    'synset': 'marker.n.03',
    'synonyms': ['marker'],
    'def': 'a writing implement for making a mark',
    'name': 'marker'
}, {
    'frequency': 'r',
    'id': 683,
    'synset': 'martini.n.01',
    'synonyms': ['martini'],
    'def': 'a cocktail made of gin (or vodka) with dry vermouth',
    'name': 'martini'
}, {
    'frequency':
        'r',
    'id':
        684,
    'synset':
        'mascot.n.01',
    'synonyms': ['mascot'],
    'def':
        'a person or animal that is adopted by a team or other group as a symbolic figure',
    'name':
        'mascot'
}, {
    'frequency': 'c',
    'id': 685,
    'synset': 'mashed_potato.n.01',
    'synonyms': ['mashed_potato'],
    'def': 'potato that has been peeled and boiled and then mashed',
    'name': 'mashed_potato'
}, {
    'frequency': 'r',
    'id': 686,
    'synset': 'masher.n.02',
    'synonyms': ['masher'],
    'def': 'a kitchen utensil used for mashing (e.g. potatoes)',
    'name': 'masher'
}, {
    'frequency': 'f',
    'id': 687,
    'synset': 'mask.n.04',
    'synonyms': ['mask', 'facemask'],
    'def': 'a protective covering worn over the face',
    'name': 'mask'
}, {
    'frequency': 'f',
    'id': 688,
    'synset': 'mast.n.01',
    'synonyms': ['mast'],
    'def': 'a vertical spar for supporting sails',
    'name': 'mast'
}, {
    'frequency':
        'c',
    'id':
        689,
    'synset':
        'mat.n.03',
    'synonyms': ['mat_(gym_equipment)', 'gym_mat'],
    'def':
        'sports equipment consisting of a piece of thick padding on the floor for gymnastics',
    'name':
        'mat_(gym_equipment)'
}, {
    'frequency': 'r',
    'id': 690,
    'synset': 'matchbox.n.01',
    'synonyms': ['matchbox'],
    'def': 'a box for holding matches',
    'name': 'matchbox'
}, {
    'frequency':
        'f',
    'id':
        691,
    'synset':
        'mattress.n.01',
    'synonyms': ['mattress'],
    'def':
        'a thick pad filled with resilient material used as a bed or part of a bed',
    'name':
        'mattress'
}, {
    'frequency': 'c',
    'id': 692,
    'synset': 'measuring_cup.n.01',
    'synonyms': ['measuring_cup'],
    'def': 'graduated cup used to measure liquid or granular ingredients',
    'name': 'measuring_cup'
}, {
    'frequency':
        'c',
    'id':
        693,
    'synset':
        'measuring_stick.n.01',
    'synonyms': ['measuring_stick', 'ruler_(measuring_stick)', 'measuring_rod'],
    'def':
        'measuring instrument having a sequence of marks at regular intervals',
    'name':
        'measuring_stick'
}, {
    'frequency': 'c',
    'id': 694,
    'synset': 'meatball.n.01',
    'synonyms': ['meatball'],
    'def': 'ground meat formed into a ball and fried or simmered in broth',
    'name': 'meatball'
}, {
    'frequency':
        'c',
    'id':
        695,
    'synset':
        'medicine.n.02',
    'synonyms': ['medicine'],
    'def':
        'something that treats or prevents or alleviates the symptoms of disease',
    'name':
        'medicine'
}, {
    'frequency': 'r',
    'id': 696,
    'synset': 'melon.n.01',
    'synonyms': ['melon'],
    'def': 'fruit of the gourd family having a hard rind and sweet juicy flesh',
    'name': 'melon'
}, {
    'frequency': 'f',
    'id': 697,
    'synset': 'microphone.n.01',
    'synonyms': ['microphone'],
    'def': 'device for converting sound waves into electrical energy',
    'name': 'microphone'
}, {
    'frequency': 'r',
    'id': 698,
    'synset': 'microscope.n.01',
    'synonyms': ['microscope'],
    'def': 'magnifier of the image of small objects',
    'name': 'microscope'
}, {
    'frequency':
        'f',
    'id':
        699,
    'synset':
        'microwave.n.02',
    'synonyms': ['microwave_oven'],
    'def':
        'kitchen appliance that cooks food by passing an electromagnetic wave through it',
    'name':
        'microwave_oven'
}, {
    'frequency': 'r',
    'id': 700,
    'synset': 'milestone.n.01',
    'synonyms': ['milestone', 'milepost'],
    'def': 'stone post at side of a road to show distances',
    'name': 'milestone'
}, {
    'frequency':
        'c',
    'id':
        701,
    'synset':
        'milk.n.01',
    'synonyms': ['milk'],
    'def':
        'a white nutritious liquid secreted by mammals and used as food by human beings',
    'name':
        'milk'
}, {
    'frequency': 'f',
    'id': 702,
    'synset': 'minivan.n.01',
    'synonyms': ['minivan'],
    'def': 'a small box-shaped passenger van',
    'name': 'minivan'
}, {
    'frequency': 'r',
    'id': 703,
    'synset': 'mint.n.05',
    'synonyms': ['mint_candy'],
    'def': 'a candy that is flavored with a mint oil',
    'name': 'mint_candy'
}, {
    'frequency': 'f',
    'id': 704,
    'synset': 'mirror.n.01',
    'synonyms': ['mirror'],
    'def': 'polished surface that forms images by reflecting light',
    'name': 'mirror'
}, {
    'frequency':
        'c',
    'id':
        705,
    'synset':
        'mitten.n.01',
    'synonyms': ['mitten'],
    'def':
        'glove that encases the thumb separately and the other four fingers together',
    'name':
        'mitten'
}, {
    'frequency': 'c',
    'id': 706,
    'synset': 'mixer.n.04',
    'synonyms': ['mixer_(kitchen_tool)', 'stand_mixer'],
    'def': 'a kitchen utensil that is used for mixing foods',
    'name': 'mixer_(kitchen_tool)'
}, {
    'frequency': 'c',
    'id': 707,
    'synset': 'money.n.03',
    'synonyms': ['money'],
    'def': 'the official currency issued by a government or national bank',
    'name': 'money'
}, {
    'frequency': 'f',
    'id': 708,
    'synset': 'monitor.n.04',
    'synonyms': ['monitor_(computer_equipment) computer_monitor'],
    'def': 'a computer monitor',
    'name': 'monitor_(computer_equipment) computer_monitor'
}, {
    'frequency': 'c',
    'id': 709,
    'synset': 'monkey.n.01',
    'synonyms': ['monkey'],
    'def': 'any of various long-tailed primates',
    'name': 'monkey'
}, {
    'frequency':
        'f',
    'id':
        710,
    'synset':
        'motor.n.01',
    'synonyms': ['motor'],
    'def':
        'machine that converts other forms of energy into mechanical energy and so imparts motion',
    'name':
        'motor'
}, {
    'frequency': 'f',
    'id': 711,
    'synset': 'motor_scooter.n.01',
    'synonyms': ['motor_scooter', 'scooter'],
    'def': 'a wheeled vehicle with small wheels and a low-powered engine',
    'name': 'motor_scooter'
}, {
    'frequency': 'r',
    'id': 712,
    'synset': 'motor_vehicle.n.01',
    'synonyms': ['motor_vehicle', 'automotive_vehicle'],
    'def': 'a self-propelled wheeled vehicle that does not run on rails',
    'name': 'motor_vehicle'
}, {
    'frequency': 'r',
    'id': 713,
    'synset': 'motorboat.n.01',
    'synonyms': ['motorboat', 'powerboat'],
    'def': 'a boat propelled by an internal-combustion engine',
    'name': 'motorboat'
}, {
    'frequency': 'f',
    'id': 714,
    'synset': 'motorcycle.n.01',
    'synonyms': ['motorcycle'],
    'def': 'a motor vehicle with two wheels and a strong frame',
    'name': 'motorcycle'
}, {
    'frequency': 'f',
    'id': 715,
    'synset': 'mound.n.01',
    'synonyms': ['mound_(baseball)', "pitcher's_mound"],
    'def': '(baseball) the slight elevation on which the pitcher stands',
    'name': 'mound_(baseball)'
}, {
    'frequency':
        'r',
    'id':
        716,
    'synset':
        'mouse.n.01',
    'synonyms': ['mouse_(animal_rodent)'],
    'def':
        'a small rodent with pointed snouts and small ears on elongated bodies with slender usually hairless tails',
    'name':
        'mouse_(animal_rodent)'
}, {
    'frequency': 'f',
    'id': 717,
    'synset': 'mouse.n.04',
    'synonyms': ['mouse_(computer_equipment)', 'computer_mouse'],
    'def': 'a computer input device that controls an on-screen pointer',
    'name': 'mouse_(computer_equipment)'
}, {
    'frequency':
        'f',
    'id':
        718,
    'synset':
        'mousepad.n.01',
    'synonyms': ['mousepad'],
    'def':
        'a small portable pad that provides an operating surface for a computer mouse',
    'name':
        'mousepad'
}, {
    'frequency': 'c',
    'id': 719,
    'synset': 'muffin.n.01',
    'synonyms': ['muffin'],
    'def': 'a sweet quick bread baked in a cup-shaped pan',
    'name': 'muffin'
}, {
    'frequency': 'f',
    'id': 720,
    'synset': 'mug.n.04',
    'synonyms': ['mug'],
    'def': 'with handle and usually cylindrical',
    'name': 'mug'
}, {
    'frequency': 'f',
    'id': 721,
    'synset': 'mushroom.n.02',
    'synonyms': ['mushroom'],
    'def': 'a common mushroom',
    'name': 'mushroom'
}, {
    'frequency': 'r',
    'id': 722,
    'synset': 'music_stool.n.01',
    'synonyms': ['music_stool', 'piano_stool'],
    'def': 'a stool for piano players; usually adjustable in height',
    'name': 'music_stool'
}, {
    'frequency':
        'r',
    'id':
        723,
    'synset':
        'musical_instrument.n.01',
    'synonyms': ['musical_instrument', 'instrument_(musical)'],
    'def':
        'any of various devices or contrivances that can be used to produce musical tones or sounds',
    'name':
        'musical_instrument'
}, {
    'frequency': 'r',
    'id': 724,
    'synset': 'nailfile.n.01',
    'synonyms': ['nailfile'],
    'def': 'a small flat file for shaping the nails',
    'name': 'nailfile'
}, {
    'frequency': 'r',
    'id': 725,
    'synset': 'nameplate.n.01',
    'synonyms': ['nameplate'],
    'def': 'a plate bearing a name',
    'name': 'nameplate'
}, {
    'frequency':
        'f',
    'id':
        726,
    'synset':
        'napkin.n.01',
    'synonyms': ['napkin', 'table_napkin', 'serviette'],
    'def':
        'a small piece of table linen or paper that is used to wipe the mouth and to cover the lap in order to protect clothing',
    'name':
        'napkin'
}, {
    'frequency': 'r',
    'id': 727,
    'synset': 'neckerchief.n.01',
    'synonyms': ['neckerchief'],
    'def': 'a kerchief worn around the neck',
    'name': 'neckerchief'
}, {
    'frequency':
        'f',
    'id':
        728,
    'synset':
        'necklace.n.01',
    'synonyms': ['necklace'],
    'def':
        'jewelry consisting of a cord or chain (often bearing gems) worn about the neck as an ornament',
    'name':
        'necklace'
}, {
    'frequency':
        'f',
    'id':
        729,
    'synset':
        'necktie.n.01',
    'synonyms': ['necktie', 'tie_(necktie)'],
    'def':
        'neckwear consisting of a long narrow piece of material worn under a collar and tied in knot at the front',
    'name':
        'necktie'
}, {
    'frequency': 'r',
    'id': 730,
    'synset': 'needle.n.03',
    'synonyms': ['needle'],
    'def': 'a sharp pointed implement (usually metal)',
    'name': 'needle'
}, {
    'frequency': 'c',
    'id': 731,
    'synset': 'nest.n.01',
    'synonyms': ['nest'],
    'def': 'a structure in which animals lay eggs or give birth to their young',
    'name': 'nest'
}, {
    'frequency': 'r',
    'id': 732,
    'synset': 'newsstand.n.01',
    'synonyms': ['newsstand'],
    'def': 'a stall where newspapers and other periodicals are sold',
    'name': 'newsstand'
}, {
    'frequency': 'c',
    'id': 733,
    'synset': 'nightwear.n.01',
    'synonyms': ['nightshirt', 'nightwear', 'sleepwear', 'nightclothes'],
    'def': 'garments designed to be worn in bed',
    'name': 'nightshirt'
}, {
    'frequency':
        'r',
    'id':
        734,
    'synset':
        'nosebag.n.01',
    'synonyms': ['nosebag_(for_animals)', 'feedbag'],
    'def':
        'a canvas bag that is used to feed an animal (such as a horse); covers the muzzle and fastens at the top of the head',
    'name':
        'nosebag_(for_animals)'
}, {
    'frequency':
        'r',
    'id':
        735,
    'synset':
        'noseband.n.01',
    'synonyms': ['noseband_(for_animals)', 'nosepiece_(for_animals)'],
    'def':
        "a strap that is the part of a bridle that goes over the animal's nose",
    'name':
        'noseband_(for_animals)'
}, {
    'frequency': 'f',
    'id': 736,
    'synset': 'notebook.n.01',
    'synonyms': ['notebook'],
    'def': 'a book with blank pages for recording notes or memoranda',
    'name': 'notebook'
}, {
    'frequency': 'c',
    'id': 737,
    'synset': 'notepad.n.01',
    'synonyms': ['notepad'],
    'def': 'a pad of paper for keeping notes',
    'name': 'notepad'
}, {
    'frequency':
        'c',
    'id':
        738,
    'synset':
        'nut.n.03',
    'synonyms': ['nut'],
    'def':
        'a small metal block (usually square or hexagonal) with internal screw thread to be fitted onto a bolt',
    'name':
        'nut'
}, {
    'frequency': 'r',
    'id': 739,
    'synset': 'nutcracker.n.01',
    'synonyms': ['nutcracker'],
    'def': 'a hand tool used to crack nuts open',
    'name': 'nutcracker'
}, {
    'frequency': 'c',
    'id': 740,
    'synset': 'oar.n.01',
    'synonyms': ['oar'],
    'def': 'an implement used to propel or steer a boat',
    'name': 'oar'
}, {
    'frequency': 'r',
    'id': 741,
    'synset': 'octopus.n.01',
    'synonyms': ['octopus_(food)'],
    'def': 'tentacles of octopus prepared as food',
    'name': 'octopus_(food)'
}, {
    'frequency':
        'r',
    'id':
        742,
    'synset':
        'octopus.n.02',
    'synonyms': ['octopus_(animal)'],
    'def':
        'bottom-living cephalopod having a soft oval body with eight long tentacles',
    'name':
        'octopus_(animal)'
}, {
    'frequency': 'c',
    'id': 743,
    'synset': 'oil_lamp.n.01',
    'synonyms': ['oil_lamp', 'kerosene_lamp', 'kerosine_lamp'],
    'def': 'a lamp that burns oil (as kerosine) for light',
    'name': 'oil_lamp'
}, {
    'frequency': 'c',
    'id': 744,
    'synset': 'olive_oil.n.01',
    'synonyms': ['olive_oil'],
    'def': 'oil from olives',
    'name': 'olive_oil'
}, {
    'frequency':
        'r',
    'id':
        745,
    'synset':
        'omelet.n.01',
    'synonyms': ['omelet', 'omelette'],
    'def':
        'beaten eggs cooked until just set; may be folded around e.g. ham or cheese or jelly',
    'name':
        'omelet'
}, {
    'frequency': 'f',
    'id': 746,
    'synset': 'onion.n.01',
    'synonyms': ['onion'],
    'def': 'the bulb of an onion plant',
    'name': 'onion'
}, {
    'frequency': 'f',
    'id': 747,
    'synset': 'orange.n.01',
    'synonyms': ['orange_(fruit)'],
    'def': 'orange (FRUIT of an orange tree)',
    'name': 'orange_(fruit)'
}, {
    'frequency': 'c',
    'id': 748,
    'synset': 'orange_juice.n.01',
    'synonyms': ['orange_juice'],
    'def': 'bottled or freshly squeezed juice of oranges',
    'name': 'orange_juice'
}, {
    'frequency': 'r',
    'id': 749,
    'synset': 'oregano.n.01',
    'synonyms': ['oregano', 'marjoram'],
    'def': 'aromatic Eurasian perennial herb used in cooking and baking',
    'name': 'oregano'
}, {
    'frequency':
        'c',
    'id':
        750,
    'synset':
        'ostrich.n.02',
    'synonyms': ['ostrich'],
    'def':
        'fast-running African flightless bird with two-toed feet; largest living bird',
    'name':
        'ostrich'
}, {
    'frequency': 'c',
    'id': 751,
    'synset': 'ottoman.n.03',
    'synonyms': ['ottoman', 'pouf', 'pouffe', 'hassock'],
    'def': 'thick cushion used as a seat',
    'name': 'ottoman'
}, {
    'frequency':
        'c',
    'id':
        752,
    'synset':
        'overall.n.01',
    'synonyms': ['overalls_(clothing)'],
    'def':
        'work clothing consisting of denim trousers usually with a bib and shoulder straps',
    'name':
        'overalls_(clothing)'
}, {
    'frequency':
        'c',
    'id':
        753,
    'synset':
        'owl.n.01',
    'synonyms': ['owl'],
    'def':
        'nocturnal bird of prey with hawk-like beak and claws and large head with front-facing eyes',
    'name':
        'owl'
}, {
    'frequency': 'c',
    'id': 754,
    'synset': 'packet.n.03',
    'synonyms': ['packet'],
    'def': 'a small package or bundle',
    'name': 'packet'
}, {
    'frequency':
        'r',
    'id':
        755,
    'synset':
        'pad.n.03',
    'synonyms': ['inkpad', 'inking_pad', 'stamp_pad'],
    'def':
        'absorbent material saturated with ink used to transfer ink evenly to a rubber stamp',
    'name':
        'inkpad'
}, {
    'frequency':
        'c',
    'id':
        756,
    'synset':
        'pad.n.04',
    'synonyms': ['pad'],
    'def':
        'a flat mass of soft material used for protection, stuffing, or comfort',
    'name':
        'pad'
}, {
    'frequency':
        'c',
    'id':
        757,
    'synset':
        'paddle.n.04',
    'synonyms': ['paddle', 'boat_paddle'],
    'def':
        'a short light oar used without an oarlock to propel a canoe or small boat',
    'name':
        'paddle'
}, {
    'frequency': 'c',
    'id': 758,
    'synset': 'padlock.n.01',
    'synonyms': ['padlock'],
    'def': 'a detachable, portable lock',
    'name': 'padlock'
}, {
    'frequency': 'r',
    'id': 759,
    'synset': 'paintbox.n.01',
    'synonyms': ['paintbox'],
    'def': "a box containing a collection of cubes or tubes of artists' paint",
    'name': 'paintbox'
}, {
    'frequency': 'c',
    'id': 760,
    'synset': 'paintbrush.n.01',
    'synonyms': ['paintbrush'],
    'def': 'a brush used as an applicator to apply paint',
    'name': 'paintbrush'
}, {
    'frequency':
        'f',
    'id':
        761,
    'synset':
        'painting.n.01',
    'synonyms': ['painting'],
    'def':
        'graphic art consisting of an artistic composition made by applying paints to a surface',
    'name':
        'painting'
}, {
    'frequency': 'c',
    'id': 762,
    'synset': 'pajama.n.02',
    'synonyms': ['pajamas', 'pyjamas'],
    'def': 'loose-fitting nightclothes worn for sleeping or lounging',
    'name': 'pajamas'
}, {
    'frequency':
        'c',
    'id':
        763,
    'synset':
        'palette.n.02',
    'synonyms': ['palette', 'pallet'],
    'def':
        'board that provides a flat surface on which artists mix paints and the range of colors used',
    'name':
        'palette'
}, {
    'frequency': 'f',
    'id': 764,
    'synset': 'pan.n.01',
    'synonyms': ['pan_(for_cooking)', 'cooking_pan'],
    'def': 'cooking utensil consisting of a wide metal vessel',
    'name': 'pan_(for_cooking)'
}, {
    'frequency': 'r',
    'id': 765,
    'synset': 'pan.n.03',
    'synonyms': ['pan_(metal_container)'],
    'def': 'shallow container made of metal',
    'name': 'pan_(metal_container)'
}, {
    'frequency': 'c',
    'id': 766,
    'synset': 'pancake.n.01',
    'synonyms': ['pancake'],
    'def': 'a flat cake of thin batter fried on both sides on a griddle',
    'name': 'pancake'
}, {
    'frequency': 'r',
    'id': 767,
    'synset': 'pantyhose.n.01',
    'synonyms': ['pantyhose'],
    'def': "a woman's tights consisting of underpants and stockings",
    'name': 'pantyhose'
}, {
    'frequency': 'r',
    'id': 768,
    'synset': 'papaya.n.02',
    'synonyms': ['papaya'],
    'def': 'large oval melon-like tropical fruit with yellowish flesh',
    'name': 'papaya'
}, {
    'frequency': 'r',
    'id': 769,
    'synset': 'paper_clip.n.01',
    'synonyms': ['paperclip'],
    'def': 'a wire or plastic clip for holding sheets of paper together',
    'name': 'paperclip'
}, {
    'frequency': 'f',
    'id': 770,
    'synset': 'paper_plate.n.01',
    'synonyms': ['paper_plate'],
    'def': 'a disposable plate made of cardboard',
    'name': 'paper_plate'
}, {
    'frequency': 'f',
    'id': 771,
    'synset': 'paper_towel.n.01',
    'synonyms': ['paper_towel'],
    'def': 'a disposable towel made of absorbent paper',
    'name': 'paper_towel'
}, {
    'frequency': 'r',
    'id': 772,
    'synset': 'paperback_book.n.01',
    'synonyms': [
        'paperback_book', 'paper-back_book', 'softback_book', 'soft-cover_book'
    ],
    'def': 'a book with paper covers',
    'name': 'paperback_book'
}, {
    'frequency': 'r',
    'id': 773,
    'synset': 'paperweight.n.01',
    'synonyms': ['paperweight'],
    'def': 'a weight used to hold down a stack of papers',
    'name': 'paperweight'
}, {
    'frequency':
        'c',
    'id':
        774,
    'synset':
        'parachute.n.01',
    'synonyms': ['parachute'],
    'def':
        'rescue equipment consisting of a device that fills with air and retards your fall',
    'name':
        'parachute'
}, {
    'frequency': 'r',
    'id': 775,
    'synset': 'parakeet.n.01',
    'synonyms': [
        'parakeet', 'parrakeet', 'parroket', 'paraquet', 'paroquet', 'parroquet'
    ],
    'def': 'any of numerous small slender long-tailed parrots',
    'name': 'parakeet'
}, {
    'frequency':
        'c',
    'id':
        776,
    'synset':
        'parasail.n.01',
    'synonyms': ['parasail_(sports)'],
    'def':
        'parachute that will lift a person up into the air when it is towed by a motorboat or a car',
    'name':
        'parasail_(sports)'
}, {
    'frequency': 'r',
    'id': 777,
    'synset': 'parchment.n.01',
    'synonyms': ['parchment'],
    'def': 'a superior paper resembling sheepskin',
    'name': 'parchment'
}, {
    'frequency': 'r',
    'id': 778,
    'synset': 'parka.n.01',
    'synonyms': ['parka', 'anorak'],
    'def': "a kind of heavy jacket (`windcheater' is a British term)",
    'name': 'parka'
}, {
    'frequency': 'f',
    'id': 779,
    'synset': 'parking_meter.n.01',
    'synonyms': ['parking_meter'],
    'def': 'a coin-operated timer located next to a parking space',
    'name': 'parking_meter'
}, {
    'frequency':
        'c',
    'id':
        780,
    'synset':
        'parrot.n.01',
    'synonyms': ['parrot'],
    'def':
        'usually brightly colored tropical birds with short hooked beaks and the ability to mimic sounds',
    'name':
        'parrot'
}, {
    'frequency': 'c',
    'id': 781,
    'synset': 'passenger_car.n.01',
    'synonyms': ['passenger_car_(part_of_a_train)', 'coach_(part_of_a_train)'],
    'def': 'a railcar where passengers ride',
    'name': 'passenger_car_(part_of_a_train)'
}, {
    'frequency': 'r',
    'id': 782,
    'synset': 'passenger_ship.n.01',
    'synonyms': ['passenger_ship'],
    'def': 'a ship built to carry passengers',
    'name': 'passenger_ship'
}, {
    'frequency':
        'r',
    'id':
        783,
    'synset':
        'passport.n.02',
    'synonyms': ['passport'],
    'def':
        'a document issued by a country to a citizen allowing that person to travel abroad and re-enter the home country',
    'name':
        'passport'
}, {
    'frequency': 'f',
    'id': 784,
    'synset': 'pastry.n.02',
    'synonyms': ['pastry'],
    'def': 'any of various baked foods made of dough or batter',
    'name': 'pastry'
}, {
    'frequency': 'r',
    'id': 785,
    'synset': 'patty.n.01',
    'synonyms': ['patty_(food)'],
    'def': 'small flat mass of chopped food',
    'name': 'patty_(food)'
}, {
    'frequency': 'c',
    'id': 786,
    'synset': 'pea.n.01',
    'synonyms': ['pea_(food)'],
    'def': 'seed of a pea plant used for food',
    'name': 'pea_(food)'
}, {
    'frequency': 'c',
    'id': 787,
    'synset': 'peach.n.03',
    'synonyms': ['peach'],
    'def': 'downy juicy fruit with sweet yellowish or whitish flesh',
    'name': 'peach'
}, {
    'frequency': 'c',
    'id': 788,
    'synset': 'peanut_butter.n.01',
    'synonyms': ['peanut_butter'],
    'def': 'a spread made from ground peanuts',
    'name': 'peanut_butter'
}, {
    'frequency': 'c',
    'id': 789,
    'synset': 'pear.n.01',
    'synonyms': ['pear'],
    'def': 'sweet juicy gritty-textured fruit available in many varieties',
    'name': 'pear'
}, {
    'frequency': 'r',
    'id': 790,
    'synset': 'peeler.n.03',
    'synonyms': ['peeler_(tool_for_fruit_and_vegetables)'],
    'def': 'a device for peeling vegetables or fruits',
    'name': 'peeler_(tool_for_fruit_and_vegetables)'
}, {
    'frequency':
        'r',
    'id':
        791,
    'synset':
        'pegboard.n.01',
    'synonyms': ['pegboard'],
    'def':
        'a board perforated with regularly spaced holes into which pegs can be fitted',
    'name':
        'pegboard'
}, {
    'frequency':
        'c',
    'id':
        792,
    'synset':
        'pelican.n.01',
    'synonyms': ['pelican'],
    'def':
        'large long-winged warm-water seabird having a large bill with a distensible pouch for fish',
    'name':
        'pelican'
}, {
    'frequency': 'f',
    'id': 793,
    'synset': 'pen.n.01',
    'synonyms': ['pen'],
    'def': 'a writing implement with a point from which ink flows',
    'name': 'pen'
}, {
    'frequency':
        'c',
    'id':
        794,
    'synset':
        'pencil.n.01',
    'synonyms': ['pencil'],
    'def':
        'a thin cylindrical pointed writing implement made of wood and graphite',
    'name':
        'pencil'
}, {
    'frequency': 'r',
    'id': 795,
    'synset': 'pencil_box.n.01',
    'synonyms': ['pencil_box', 'pencil_case'],
    'def': 'a box for holding pencils',
    'name': 'pencil_box'
}, {
    'frequency': 'r',
    'id': 796,
    'synset': 'pencil_sharpener.n.01',
    'synonyms': ['pencil_sharpener'],
    'def': 'a rotary implement for sharpening the point on pencils',
    'name': 'pencil_sharpener'
}, {
    'frequency':
        'r',
    'id':
        797,
    'synset':
        'pendulum.n.01',
    'synonyms': ['pendulum'],
    'def':
        'an apparatus consisting of an object mounted so that it swings freely under the influence of gravity',
    'name':
        'pendulum'
}, {
    'frequency':
        'c',
    'id':
        798,
    'synset':
        'penguin.n.01',
    'synonyms': ['penguin'],
    'def':
        'short-legged flightless birds of cold southern regions having webbed feet and wings modified as flippers',
    'name':
        'penguin'
}, {
    'frequency': 'r',
    'id': 799,
    'synset': 'pennant.n.02',
    'synonyms': ['pennant'],
    'def': 'a flag longer than it is wide (and often tapering)',
    'name': 'pennant'
}, {
    'frequency': 'r',
    'id': 800,
    'synset': 'penny.n.02',
    'synonyms': ['penny_(coin)'],
    'def': 'a coin worth one-hundredth of the value of the basic unit',
    'name': 'penny_(coin)'
}, {
    'frequency':
        'c',
    'id':
        801,
    'synset':
        'pepper.n.03',
    'synonyms': ['pepper', 'peppercorn'],
    'def':
        'pungent seasoning from the berry of the common pepper plant; whole or ground',
    'name':
        'pepper'
}, {
    'frequency': 'c',
    'id': 802,
    'synset': 'pepper_mill.n.01',
    'synonyms': ['pepper_mill', 'pepper_grinder'],
    'def': 'a mill for grinding pepper',
    'name': 'pepper_mill'
}, {
    'frequency': 'c',
    'id': 803,
    'synset': 'perfume.n.02',
    'synonyms': ['perfume'],
    'def': 'a toiletry that emits and diffuses a fragrant odor',
    'name': 'perfume'
}, {
    'frequency': 'r',
    'id': 804,
    'synset': 'persimmon.n.02',
    'synonyms': ['persimmon'],
    'def': 'orange fruit resembling a plum; edible when fully ripe',
    'name': 'persimmon'
}, {
    'frequency': 'f',
    'id': 805,
    'synset': 'person.n.01',
    'synonyms': [
        'baby', 'child', 'boy', 'girl', 'man', 'woman', 'person', 'human'
    ],
    'def': 'a human being',
    'name': 'baby'
}, {
    'frequency': 'r',
    'id': 806,
    'synset': 'pet.n.01',
    'synonyms': ['pet'],
    'def': 'a domesticated animal kept for companionship or amusement',
    'name': 'pet'
}, {
    'frequency': 'r',
    'id': 807,
    'synset': 'petfood.n.01',
    'synonyms': ['petfood', 'pet-food'],
    'def': 'food prepared for animal pets',
    'name': 'petfood'
}, {
    'frequency': 'r',
    'id': 808,
    'synset': 'pew.n.01',
    'synonyms': ['pew_(church_bench)', 'church_bench'],
    'def': 'long bench with backs; used in church by the congregation',
    'name': 'pew_(church_bench)'
}, {
    'frequency':
        'r',
    'id':
        809,
    'synset':
        'phonebook.n.01',
    'synonyms': ['phonebook', 'telephone_book', 'telephone_directory'],
    'def':
        'a directory containing an alphabetical list of telephone subscribers and their telephone numbers',
    'name':
        'phonebook'
}, {
    'frequency':
        'c',
    'id':
        810,
    'synset':
        'phonograph_record.n.01',
    'synonyms': [
        'phonograph_record', 'phonograph_recording',
        'record_(phonograph_recording)'
    ],
    'def':
        'sound recording consisting of a typically black disk with a continuous groove',
    'name':
        'phonograph_record'
}, {
    'frequency':
        'c',
    'id':
        811,
    'synset':
        'piano.n.01',
    'synonyms': ['piano'],
    'def':
        'a keyboard instrument that is played by depressing keys that cause hammers to strike tuned strings and produce sounds',
    'name':
        'piano'
}, {
    'frequency': 'f',
    'id': 812,
    'synset': 'pickle.n.01',
    'synonyms': ['pickle'],
    'def': 'vegetables (especially cucumbers) preserved in brine or vinegar',
    'name': 'pickle'
}, {
    'frequency': 'f',
    'id': 813,
    'synset': 'pickup.n.01',
    'synonyms': ['pickup_truck'],
    'def': 'a light truck with an open body and low sides and a tailboard',
    'name': 'pickup_truck'
}, {
    'frequency': 'c',
    'id': 814,
    'synset': 'pie.n.01',
    'synonyms': ['pie'],
    'def': 'dish baked in pastry-lined pan often with a pastry top',
    'name': 'pie'
}, {
    'frequency': 'c',
    'id': 815,
    'synset': 'pigeon.n.01',
    'synonyms': ['pigeon'],
    'def': 'wild and domesticated birds having a heavy body and short legs',
    'name': 'pigeon'
}, {
    'frequency': 'r',
    'id': 816,
    'synset': 'piggy_bank.n.01',
    'synonyms': ['piggy_bank', 'penny_bank'],
    'def': "a child's coin bank (often shaped like a pig)",
    'name': 'piggy_bank'
}, {
    'frequency': 'f',
    'id': 817,
    'synset': 'pillow.n.01',
    'synonyms': ['pillow'],
    'def': 'a cushion to support the head of a sleeping person',
    'name': 'pillow'
}, {
    'frequency':
        'r',
    'id':
        818,
    'synset':
        'pin.n.09',
    'synonyms': ['pin_(non_jewelry)'],
    'def':
        'a small slender (often pointed) piece of wood or metal used to support or fasten or attach things',
    'name':
        'pin_(non_jewelry)'
}, {
    'frequency': 'f',
    'id': 819,
    'synset': 'pineapple.n.02',
    'synonyms': ['pineapple'],
    'def': 'large sweet fleshy tropical fruit with a tuft of stiff leaves',
    'name': 'pineapple'
}, {
    'frequency': 'c',
    'id': 820,
    'synset': 'pinecone.n.01',
    'synonyms': ['pinecone'],
    'def': 'the seed-producing cone of a pine tree',
    'name': 'pinecone'
}, {
    'frequency': 'r',
    'id': 821,
    'synset': 'ping-pong_ball.n.01',
    'synonyms': ['ping-pong_ball'],
    'def': 'light hollow ball used in playing table tennis',
    'name': 'ping-pong_ball'
}, {
    'frequency':
        'r',
    'id':
        822,
    'synset':
        'pinwheel.n.03',
    'synonyms': ['pinwheel'],
    'def':
        'a toy consisting of vanes of colored paper or plastic that is pinned to a stick and spins when it is pointed into the wind',
    'name':
        'pinwheel'
}, {
    'frequency': 'r',
    'id': 823,
    'synset': 'pipe.n.01',
    'synonyms': ['tobacco_pipe'],
    'def': 'a tube with a small bowl at one end; used for smoking tobacco',
    'name': 'tobacco_pipe'
}, {
    'frequency':
        'f',
    'id':
        824,
    'synset':
        'pipe.n.02',
    'synonyms': ['pipe', 'piping'],
    'def':
        'a long tube made of metal or plastic that is used to carry water or oil or gas etc.',
    'name':
        'pipe'
}, {
    'frequency': 'r',
    'id': 825,
    'synset': 'pistol.n.01',
    'synonyms': ['pistol', 'handgun'],
    'def': 'a firearm that is held and fired with one hand',
    'name': 'pistol'
}, {
    'frequency': 'r',
    'id': 826,
    'synset': 'pita.n.01',
    'synonyms': ['pita_(bread)', 'pocket_bread'],
    'def': 'usually small round bread that can open into a pocket for filling',
    'name': 'pita_(bread)'
}, {
    'frequency': 'f',
    'id': 827,
    'synset': 'pitcher.n.02',
    'synonyms': ['pitcher_(vessel_for_liquid)', 'ewer'],
    'def': 'an open vessel with a handle and a spout for pouring',
    'name': 'pitcher_(vessel_for_liquid)'
}, {
    'frequency':
        'r',
    'id':
        828,
    'synset':
        'pitchfork.n.01',
    'synonyms': ['pitchfork'],
    'def':
        'a long-handled hand tool with sharp widely spaced prongs for lifting and pitching hay',
    'name':
        'pitchfork'
}, {
    'frequency':
        'f',
    'id':
        829,
    'synset':
        'pizza.n.01',
    'synonyms': ['pizza'],
    'def':
        'Italian open pie made of thin bread dough spread with a spiced mixture of e.g. tomato sauce and cheese',
    'name':
        'pizza'
}, {
    'frequency': 'f',
    'id': 830,
    'synset': 'place_mat.n.01',
    'synonyms': ['place_mat'],
    'def': 'a mat placed on a table for an individual place setting',
    'name': 'place_mat'
}, {
    'frequency': 'f',
    'id': 831,
    'synset': 'plate.n.04',
    'synonyms': ['plate'],
    'def': 'dish on which food is served or from which food is eaten',
    'name': 'plate'
}, {
    'frequency': 'c',
    'id': 832,
    'synset': 'platter.n.01',
    'synonyms': ['platter'],
    'def': 'a large shallow dish used for serving food',
    'name': 'platter'
}, {
    'frequency': 'r',
    'id': 833,
    'synset': 'playing_card.n.01',
    'synonyms': ['playing_card'],
    'def': 'one of a pack of cards that are used to play card games',
    'name': 'playing_card'
}, {
    'frequency': 'r',
    'id': 834,
    'synset': 'playpen.n.01',
    'synonyms': ['playpen'],
    'def': 'a portable enclosure in which babies may be left to play',
    'name': 'playpen'
}, {
    'frequency':
        'c',
    'id':
        835,
    'synset':
        'pliers.n.01',
    'synonyms': ['pliers', 'plyers'],
    'def':
        'a gripping hand tool with two hinged arms and (usually) serrated jaws',
    'name':
        'pliers'
}, {
    'frequency':
        'r',
    'id':
        836,
    'synset':
        'plow.n.01',
    'synonyms': ['plow_(farm_equipment)', 'plough_(farm_equipment)'],
    'def':
        'a farm tool having one or more heavy blades to break the soil and cut a furrow prior to sowing',
    'name':
        'plow_(farm_equipment)'
}, {
    'frequency': 'r',
    'id': 837,
    'synset': 'pocket_watch.n.01',
    'synonyms': ['pocket_watch'],
    'def': 'a watch that is carried in a small watch pocket',
    'name': 'pocket_watch'
}, {
    'frequency':
        'c',
    'id':
        838,
    'synset':
        'pocketknife.n.01',
    'synonyms': ['pocketknife'],
    'def':
        'a knife with a blade that folds into the handle; suitable for carrying in the pocket',
    'name':
        'pocketknife'
}, {
    'frequency':
        'c',
    'id':
        839,
    'synset':
        'poker.n.01',
    'synonyms': ['poker_(fire_stirring_tool)', 'stove_poker', 'fire_hook'],
    'def':
        'fire iron consisting of a metal rod with a handle; used to stir a fire',
    'name':
        'poker_(fire_stirring_tool)'
}, {
    'frequency': 'f',
    'id': 840,
    'synset': 'pole.n.01',
    'synonyms': ['pole', 'post'],
    'def': 'a long (usually round) rod of wood or metal or plastic',
    'name': 'pole'
}, {
    'frequency': 'r',
    'id': 841,
    'synset': 'police_van.n.01',
    'synonyms': ['police_van', 'police_wagon', 'paddy_wagon', 'patrol_wagon'],
    'def': 'van used by police to transport prisoners',
    'name': 'police_van'
}, {
    'frequency': 'f',
    'id': 842,
    'synset': 'polo_shirt.n.01',
    'synonyms': ['polo_shirt', 'sport_shirt'],
    'def': 'a shirt with short sleeves designed for comfort and casual wear',
    'name': 'polo_shirt'
}, {
    'frequency': 'r',
    'id': 843,
    'synset': 'poncho.n.01',
    'synonyms': ['poncho'],
    'def': 'a blanket-like cloak with a hole in the center for the head',
    'name': 'poncho'
}, {
    'frequency':
        'c',
    'id':
        844,
    'synset':
        'pony.n.05',
    'synonyms': ['pony'],
    'def':
        'any of various breeds of small gentle horses usually less than five feet high at the shoulder',
    'name':
        'pony'
}, {
    'frequency': 'r',
    'id': 845,
    'synset': 'pool_table.n.01',
    'synonyms': ['pool_table', 'billiard_table', 'snooker_table'],
    'def': 'game equipment consisting of a heavy table on which pool is played',
    'name': 'pool_table'
}, {
    'frequency': 'f',
    'id': 846,
    'synset': 'pop.n.02',
    'synonyms': ['pop_(soda)', 'soda_(pop)', 'tonic', 'soft_drink'],
    'def': 'a sweet drink containing carbonated water and flavoring',
    'name': 'pop_(soda)'
}, {
    'frequency': 'r',
    'id': 847,
    'synset': 'portrait.n.02',
    'synonyms': ['portrait', 'portrayal'],
    'def': 'any likeness of a person, in any medium',
    'name': 'portrait'
}, {
    'frequency': 'c',
    'id': 848,
    'synset': 'postbox.n.01',
    'synonyms': ['postbox_(public)', 'mailbox_(public)'],
    'def': 'public box for deposit of mail',
    'name': 'postbox_(public)'
}, {
    'frequency': 'c',
    'id': 849,
    'synset': 'postcard.n.01',
    'synonyms': ['postcard', 'postal_card', 'mailing-card'],
    'def': 'a card for sending messages by post without an envelope',
    'name': 'postcard'
}, {
    'frequency': 'f',
    'id': 850,
    'synset': 'poster.n.01',
    'synonyms': ['poster', 'placard'],
    'def': 'a sign posted in a public place as an advertisement',
    'name': 'poster'
}, {
    'frequency':
        'f',
    'id':
        851,
    'synset':
        'pot.n.01',
    'synonyms': ['pot'],
    'def':
        'metal or earthenware cooking vessel that is usually round and deep; often has a handle and lid',
    'name':
        'pot'
}, {
    'frequency': 'f',
    'id': 852,
    'synset': 'pot.n.04',
    'synonyms': ['flowerpot'],
    'def': 'a container in which plants are cultivated',
    'name': 'flowerpot'
}, {
    'frequency': 'f',
    'id': 853,
    'synset': 'potato.n.01',
    'synonyms': ['potato'],
    'def': 'an edible tuber native to South America',
    'name': 'potato'
}, {
    'frequency': 'c',
    'id': 854,
    'synset': 'potholder.n.01',
    'synonyms': ['potholder'],
    'def': 'an insulated pad for holding hot pots',
    'name': 'potholder'
}, {
    'frequency': 'c',
    'id': 855,
    'synset': 'pottery.n.01',
    'synonyms': ['pottery', 'clayware'],
    'def': 'ceramic ware made from clay and baked in a kiln',
    'name': 'pottery'
}, {
    'frequency': 'c',
    'id': 856,
    'synset': 'pouch.n.01',
    'synonyms': ['pouch'],
    'def': 'a small or medium size container for holding or carrying things',
    'name': 'pouch'
}, {
    'frequency': 'r',
    'id': 857,
    'synset': 'power_shovel.n.01',
    'synonyms': ['power_shovel', 'excavator', 'digger'],
    'def': 'a machine for excavating',
    'name': 'power_shovel'
}, {
    'frequency': 'c',
    'id': 858,
    'synset': 'prawn.n.01',
    'synonyms': ['prawn', 'shrimp'],
    'def': 'any of various edible decapod crustaceans',
    'name': 'prawn'
}, {
    'frequency': 'f',
    'id': 859,
    'synset': 'printer.n.03',
    'synonyms': ['printer', 'printing_machine'],
    'def': 'a machine that prints',
    'name': 'printer'
}, {
    'frequency': 'c',
    'id': 860,
    'synset': 'projectile.n.01',
    'synonyms': ['projectile_(weapon)', 'missile'],
    'def': 'a weapon that is forcibly thrown or projected at a targets',
    'name': 'projectile_(weapon)'
}, {
    'frequency':
        'c',
    'id':
        861,
    'synset':
        'projector.n.02',
    'synonyms': ['projector'],
    'def':
        'an optical instrument that projects an enlarged image onto a screen',
    'name':
        'projector'
}, {
    'frequency': 'f',
    'id': 862,
    'synset': 'propeller.n.01',
    'synonyms': ['propeller', 'propellor'],
    'def': 'a mechanical device that rotates to push against air or water',
    'name': 'propeller'
}, {
    'frequency': 'r',
    'id': 863,
    'synset': 'prune.n.01',
    'synonyms': ['prune'],
    'def': 'dried plum',
    'name': 'prune'
}, {
    'frequency': 'r',
    'id': 864,
    'synset': 'pudding.n.01',
    'synonyms': ['pudding'],
    'def': 'any of various soft thick unsweetened baked dishes',
    'name': 'pudding'
}, {
    'frequency':
        'r',
    'id':
        865,
    'synset':
        'puffer.n.02',
    'synonyms': ['puffer_(fish)', 'pufferfish', 'blowfish', 'globefish'],
    'def':
        'fishes whose elongated spiny body can inflate itself with water or air to form a globe',
    'name':
        'puffer_(fish)'
}, {
    'frequency': 'r',
    'id': 866,
    'synset': 'puffin.n.01',
    'synonyms': ['puffin'],
    'def': 'seabirds having short necks and brightly colored compressed bills',
    'name': 'puffin'
}, {
    'frequency':
        'r',
    'id':
        867,
    'synset':
        'pug.n.01',
    'synonyms': ['pug-dog'],
    'def':
        'small compact smooth-coated breed of Asiatic origin having a tightly curled tail and broad flat wrinkled muzzle',
    'name':
        'pug-dog'
}, {
    'frequency':
        'c',
    'id':
        868,
    'synset':
        'pumpkin.n.02',
    'synonyms': ['pumpkin'],
    'def':
        'usually large pulpy deep-yellow round fruit of the squash family maturing in late summer or early autumn',
    'name':
        'pumpkin'
}, {
    'frequency': 'r',
    'id': 869,
    'synset': 'punch.n.03',
    'synonyms': ['puncher'],
    'def': 'a tool for making holes or indentations',
    'name': 'puncher'
}, {
    'frequency':
        'r',
    'id':
        870,
    'synset':
        'puppet.n.01',
    'synonyms': ['puppet', 'marionette'],
    'def':
        'a small figure of a person operated from above with strings by a puppeteer',
    'name':
        'puppet'
}, {
    'frequency': 'r',
    'id': 871,
    'synset': 'puppy.n.01',
    'synonyms': ['puppy'],
    'def': 'a young dog',
    'name': 'puppy'
}, {
    'frequency': 'r',
    'id': 872,
    'synset': 'quesadilla.n.01',
    'synonyms': ['quesadilla'],
    'def': 'a tortilla that is filled with cheese and heated',
    'name': 'quesadilla'
}, {
    'frequency':
        'r',
    'id':
        873,
    'synset':
        'quiche.n.02',
    'synonyms': ['quiche'],
    'def':
        'a tart filled with rich unsweetened custard; often contains other ingredients (as cheese or ham or seafood or vegetables)',
    'name':
        'quiche'
}, {
    'frequency':
        'f',
    'id':
        874,
    'synset':
        'quilt.n.01',
    'synonyms': ['quilt', 'comforter'],
    'def':
        'bedding made of two layers of cloth filled with stuffing and stitched together',
    'name':
        'quilt'
}, {
    'frequency':
        'c',
    'id':
        875,
    'synset':
        'rabbit.n.01',
    'synonyms': ['rabbit'],
    'def':
        'any of various burrowing animals of the family Leporidae having long ears and short tails',
    'name':
        'rabbit'
}, {
    'frequency': 'r',
    'id': 876,
    'synset': 'racer.n.02',
    'synonyms': ['race_car', 'racing_car'],
    'def': 'a fast car that competes in races',
    'name': 'race_car'
}, {
    'frequency': 'c',
    'id': 877,
    'synset': 'racket.n.04',
    'synonyms': ['racket', 'racquet'],
    'def': 'a sports implement used to strike a ball in various games',
    'name': 'racket'
}, {
    'frequency':
        'r',
    'id':
        878,
    'synset':
        'radar.n.01',
    'synonyms': ['radar'],
    'def':
        'measuring instrument in which the echo of a pulse of microwave radiation is used to detect and locate distant objects',
    'name':
        'radar'
}, {
    'frequency':
        'c',
    'id':
        879,
    'synset':
        'radiator.n.03',
    'synonyms': ['radiator'],
    'def':
        'a mechanism consisting of a metal honeycomb through which hot fluids circulate',
    'name':
        'radiator'
}, {
    'frequency':
        'c',
    'id':
        880,
    'synset':
        'radio_receiver.n.01',
    'synonyms': ['radio_receiver', 'radio_set', 'radio', 'tuner_(radio)'],
    'def':
        'an electronic receiver that detects and demodulates and amplifies transmitted radio signals',
    'name':
        'radio_receiver'
}, {
    'frequency': 'c',
    'id': 881,
    'synset': 'radish.n.03',
    'synonyms': ['radish', 'daikon'],
    'def': 'pungent edible root of any of various cultivated radish plants',
    'name': 'radish'
}, {
    'frequency':
        'c',
    'id':
        882,
    'synset':
        'raft.n.01',
    'synonyms': ['raft'],
    'def':
        'a flat float (usually made of logs or planks) that can be used for transport or as a platform for swimmers',
    'name':
        'raft'
}, {
    'frequency': 'r',
    'id': 883,
    'synset': 'rag_doll.n.01',
    'synonyms': ['rag_doll'],
    'def': 'a cloth doll that is stuffed and (usually) painted',
    'name': 'rag_doll'
}, {
    'frequency': 'c',
    'id': 884,
    'synset': 'raincoat.n.01',
    'synonyms': ['raincoat', 'waterproof_jacket'],
    'def': 'a water-resistant coat',
    'name': 'raincoat'
}, {
    'frequency': 'c',
    'id': 885,
    'synset': 'ram.n.05',
    'synonyms': ['ram_(animal)'],
    'def': 'uncastrated adult male sheep',
    'name': 'ram_(animal)'
}, {
    'frequency':
        'c',
    'id':
        886,
    'synset':
        'raspberry.n.02',
    'synonyms': ['raspberry'],
    'def':
        'red or black edible aggregate berries usually smaller than the related blackberries',
    'name':
        'raspberry'
}, {
    'frequency':
        'r',
    'id':
        887,
    'synset':
        'rat.n.01',
    'synonyms': ['rat'],
    'def':
        'any of various long-tailed rodents similar to but larger than a mouse',
    'name':
        'rat'
}, {
    'frequency': 'c',
    'id': 888,
    'synset': 'razorblade.n.01',
    'synonyms': ['razorblade'],
    'def': 'a blade that has very sharp edge',
    'name': 'razorblade'
}, {
    'frequency':
        'c',
    'id':
        889,
    'synset':
        'reamer.n.01',
    'synonyms': ['reamer_(juicer)', 'juicer', 'juice_reamer'],
    'def':
        'a squeezer with a conical ridged center that is used for squeezing juice from citrus fruit',
    'name':
        'reamer_(juicer)'
}, {
    'frequency': 'f',
    'id': 890,
    'synset': 'rearview_mirror.n.01',
    'synonyms': ['rearview_mirror'],
    'def': 'car mirror that reflects the view out of the rear window',
    'name': 'rearview_mirror'
}, {
    'frequency': 'c',
    'id': 891,
    'synset': 'receipt.n.02',
    'synonyms': ['receipt'],
    'def': 'an acknowledgment (usually tangible) that payment has been made',
    'name': 'receipt'
}, {
    'frequency':
        'c',
    'id':
        892,
    'synset':
        'recliner.n.01',
    'synonyms': ['recliner', 'reclining_chair', 'lounger_(chair)'],
    'def':
        'an armchair whose back can be lowered and foot can be raised to allow the sitter to recline in it',
    'name':
        'recliner'
}, {
    'frequency':
        'r',
    'id':
        893,
    'synset':
        'record_player.n.01',
    'synonyms': ['record_player', 'phonograph_(record_player)', 'turntable'],
    'def':
        'machine in which rotating records cause a stylus to vibrate and the vibrations are amplified acoustically or electronically',
    'name':
        'record_player'
}, {
    'frequency': 'r',
    'id': 894,
    'synset': 'red_cabbage.n.02',
    'synonyms': ['red_cabbage'],
    'def': 'compact head of purplish-red leaves',
    'name': 'red_cabbage'
}, {
    'frequency': 'f',
    'id': 895,
    'synset': 'reflector.n.01',
    'synonyms': ['reflector'],
    'def': 'device that reflects light, radiation, etc.',
    'name': 'reflector'
}, {
    'frequency':
        'f',
    'id':
        896,
    'synset':
        'remote_control.n.01',
    'synonyms': ['remote_control'],
    'def':
        'a device that can be used to control a machine or apparatus from a distance',
    'name':
        'remote_control'
}, {
    'frequency':
        'c',
    'id':
        897,
    'synset':
        'rhinoceros.n.01',
    'synonyms': ['rhinoceros'],
    'def':
        'massive powerful herbivorous odd-toed ungulate of southeast Asia and Africa having very thick skin and one or two horns on the snout',
    'name':
        'rhinoceros'
}, {
    'frequency': 'r',
    'id': 898,
    'synset': 'rib.n.03',
    'synonyms': ['rib_(food)'],
    'def': 'cut of meat including one or more ribs',
    'name': 'rib_(food)'
}, {
    'frequency': 'r',
    'id': 899,
    'synset': 'rifle.n.01',
    'synonyms': ['rifle'],
    'def': 'a shoulder firearm with a long barrel',
    'name': 'rifle'
}, {
    'frequency':
        'f',
    'id':
        900,
    'synset':
        'ring.n.08',
    'synonyms': ['ring'],
    'def':
        'jewelry consisting of a circlet of precious metal (often set with jewels) worn on the finger',
    'name':
        'ring'
}, {
    'frequency': 'r',
    'id': 901,
    'synset': 'river_boat.n.01',
    'synonyms': ['river_boat'],
    'def': 'a boat used on rivers or to ply a river',
    'name': 'river_boat'
}, {
    'frequency': 'r',
    'id': 902,
    'synset': 'road_map.n.02',
    'synonyms': ['road_map'],
    'def': '(NOT A ROAD) a MAP showing roads (for automobile travel)',
    'name': 'road_map'
}, {
    'frequency': 'c',
    'id': 903,
    'synset': 'robe.n.01',
    'synonyms': ['robe'],
    'def': 'any loose flowing garment',
    'name': 'robe'
}, {
    'frequency': 'c',
    'id': 904,
    'synset': 'rocking_chair.n.01',
    'synonyms': ['rocking_chair'],
    'def': 'a chair mounted on rockers',
    'name': 'rocking_chair'
}, {
    'frequency': 'r',
    'id': 905,
    'synset': 'roller_skate.n.01',
    'synonyms': ['roller_skate'],
    'def': 'a shoe with pairs of rollers (small hard wheels) fixed to the sole',
    'name': 'roller_skate'
}, {
    'frequency': 'r',
    'id': 906,
    'synset': 'rollerblade.n.01',
    'synonyms': ['Rollerblade'],
    'def': 'an in-line variant of a roller skate',
    'name': 'Rollerblade'
}, {
    'frequency':
        'c',
    'id':
        907,
    'synset':
        'rolling_pin.n.01',
    'synonyms': ['rolling_pin'],
    'def':
        'utensil consisting of a cylinder (usually of wood) with a handle at each end; used to roll out dough',
    'name':
        'rolling_pin'
}, {
    'frequency': 'r',
    'id': 908,
    'synset': 'root_beer.n.01',
    'synonyms': ['root_beer'],
    'def': 'carbonated drink containing extracts of roots and herbs',
    'name': 'root_beer'
}, {
    'frequency': 'c',
    'id': 909,
    'synset': 'router.n.02',
    'synonyms': ['router_(computer_equipment)'],
    'def': 'a device that forwards data packets between computer networks',
    'name': 'router_(computer_equipment)'
}, {
    'frequency':
        'f',
    'id':
        910,
    'synset':
        'rubber_band.n.01',
    'synonyms': ['rubber_band', 'elastic_band'],
    'def':
        'a narrow band of elastic rubber used to hold things (such as papers) together',
    'name':
        'rubber_band'
}, {
    'frequency': 'c',
    'id': 911,
    'synset': 'runner.n.08',
    'synonyms': ['runner_(carpet)'],
    'def': 'a long narrow carpet',
    'name': 'runner_(carpet)'
}, {
    'frequency': 'f',
    'id': 912,
    'synset': 'sack.n.01',
    'synonyms': ['plastic_bag', 'paper_bag'],
    'def': "a bag made of paper or plastic for holding customer's purchases",
    'name': 'plastic_bag'
}, {
    'frequency': 'f',
    'id': 913,
    'synset': 'saddle.n.01',
    'synonyms': ['saddle_(on_an_animal)'],
    'def': 'a seat for the rider of a horse or camel',
    'name': 'saddle_(on_an_animal)'
}, {
    'frequency': 'f',
    'id': 914,
    'synset': 'saddle_blanket.n.01',
    'synonyms': ['saddle_blanket', 'saddlecloth', 'horse_blanket'],
    'def': 'stable gear consisting of a blanket placed under the saddle',
    'name': 'saddle_blanket'
}, {
    'frequency': 'c',
    'id': 915,
    'synset': 'saddlebag.n.01',
    'synonyms': ['saddlebag'],
    'def': 'a large bag (or pair of bags) hung over a saddle',
    'name': 'saddlebag'
}, {
    'frequency':
        'r',
    'id':
        916,
    'synset':
        'safety_pin.n.01',
    'synonyms': ['safety_pin'],
    'def':
        'a pin in the form of a clasp; has a guard so the point of the pin will not stick the user',
    'name':
        'safety_pin'
}, {
    'frequency':
        'c',
    'id':
        917,
    'synset':
        'sail.n.01',
    'synonyms': ['sail'],
    'def':
        'a large piece of fabric by means of which wind is used to propel a sailing vessel',
    'name':
        'sail'
}, {
    'frequency':
        'c',
    'id':
        918,
    'synset':
        'salad.n.01',
    'synonyms': ['salad'],
    'def':
        'food mixtures either arranged on a plate or tossed and served with a moist dressing; usually consisting of or including greens',
    'name':
        'salad'
}, {
    'frequency': 'r',
    'id': 919,
    'synset': 'salad_plate.n.01',
    'synonyms': ['salad_plate', 'salad_bowl'],
    'def': 'a plate or bowl for individual servings of salad',
    'name': 'salad_plate'
}, {
    'frequency': 'r',
    'id': 920,
    'synset': 'salami.n.01',
    'synonyms': ['salami'],
    'def': 'highly seasoned fatty sausage of pork and beef usually dried',
    'name': 'salami'
}, {
    'frequency': 'r',
    'id': 921,
    'synset': 'salmon.n.01',
    'synonyms': ['salmon_(fish)'],
    'def': 'any of various large food and game fishes of northern waters',
    'name': 'salmon_(fish)'
}, {
    'frequency':
        'r',
    'id':
        922,
    'synset':
        'salmon.n.03',
    'synonyms': ['salmon_(food)'],
    'def':
        'flesh of any of various marine or freshwater fish of the family Salmonidae',
    'name':
        'salmon_(food)'
}, {
    'frequency':
        'r',
    'id':
        923,
    'synset':
        'salsa.n.01',
    'synonyms': ['salsa'],
    'def':
        'spicy sauce of tomatoes and onions and chili peppers to accompany Mexican foods',
    'name':
        'salsa'
}, {
    'frequency': 'f',
    'id': 924,
    'synset': 'saltshaker.n.01',
    'synonyms': ['saltshaker'],
    'def': 'a shaker with a perforated top for sprinkling salt',
    'name': 'saltshaker'
}, {
    'frequency': 'f',
    'id': 925,
    'synset': 'sandal.n.01',
    'synonyms': ['sandal_(type_of_shoe)'],
    'def': 'a shoe consisting of a sole fastened by straps to the foot',
    'name': 'sandal_(type_of_shoe)'
}, {
    'frequency': 'f',
    'id': 926,
    'synset': 'sandwich.n.01',
    'synonyms': ['sandwich'],
    'def': 'two (or more) slices of bread with a filling between them',
    'name': 'sandwich'
}, {
    'frequency':
        'r',
    'id':
        927,
    'synset':
        'satchel.n.01',
    'synonyms': ['satchel'],
    'def':
        'luggage consisting of a small case with a flat bottom and (usually) a shoulder strap',
    'name':
        'satchel'
}, {
    'frequency': 'r',
    'id': 928,
    'synset': 'saucepan.n.01',
    'synonyms': ['saucepan'],
    'def': 'a deep pan with a handle; used for stewing or boiling',
    'name': 'saucepan'
}, {
    'frequency': 'f',
    'id': 929,
    'synset': 'saucer.n.02',
    'synonyms': ['saucer'],
    'def': 'a small shallow dish for holding a cup at the table',
    'name': 'saucer'
}, {
    'frequency': 'f',
    'id': 930,
    'synset': 'sausage.n.01',
    'synonyms': ['sausage'],
    'def': 'highly seasoned minced meat stuffed in casings',
    'name': 'sausage'
}, {
    'frequency': 'r',
    'id': 931,
    'synset': 'sawhorse.n.01',
    'synonyms': ['sawhorse', 'sawbuck'],
    'def': 'a framework for holding wood that is being sawed',
    'name': 'sawhorse'
}, {
    'frequency': 'r',
    'id': 932,
    'synset': 'sax.n.02',
    'synonyms': ['saxophone'],
    'def': "a wind instrument with a `J'-shaped form typically made of brass",
    'name': 'saxophone'
}, {
    'frequency': 'f',
    'id': 933,
    'synset': 'scale.n.07',
    'synonyms': ['scale_(measuring_instrument)'],
    'def': 'a measuring instrument for weighing; shows amount of mass',
    'name': 'scale_(measuring_instrument)'
}, {
    'frequency': 'r',
    'id': 934,
    'synset': 'scarecrow.n.01',
    'synonyms': ['scarecrow', 'strawman'],
    'def': 'an effigy in the shape of a man to frighten birds away from seeds',
    'name': 'scarecrow'
}, {
    'frequency':
        'f',
    'id':
        935,
    'synset':
        'scarf.n.01',
    'synonyms': ['scarf'],
    'def':
        'a garment worn around the head or neck or shoulders for warmth or decoration',
    'name':
        'scarf'
}, {
    'frequency': 'c',
    'id': 936,
    'synset': 'school_bus.n.01',
    'synonyms': ['school_bus'],
    'def': 'a bus used to transport children to or from school',
    'name': 'school_bus'
}, {
    'frequency': 'f',
    'id': 937,
    'synset': 'scissors.n.01',
    'synonyms': ['scissors'],
    'def': 'a tool having two crossed pivoting blades with looped handles',
    'name': 'scissors'
}, {
    'frequency':
        'c',
    'id':
        938,
    'synset':
        'scoreboard.n.01',
    'synonyms': ['scoreboard'],
    'def':
        'a large board for displaying the score of a contest (and some other information)',
    'name':
        'scoreboard'
}, {
    'frequency': 'c',
    'id': 939,
    'synset': 'scrambled_eggs.n.01',
    'synonyms': ['scrambled_eggs'],
    'def': 'eggs beaten and cooked to a soft firm consistency while stirring',
    'name': 'scrambled_eggs'
}, {
    'frequency': 'r',
    'id': 940,
    'synset': 'scraper.n.01',
    'synonyms': ['scraper'],
    'def': 'any of various hand tools for scraping',
    'name': 'scraper'
}, {
    'frequency': 'r',
    'id': 941,
    'synset': 'scratcher.n.03',
    'synonyms': ['scratcher'],
    'def': 'a device used for scratching',
    'name': 'scratcher'
}, {
    'frequency':
        'c',
    'id':
        942,
    'synset':
        'screwdriver.n.01',
    'synonyms': ['screwdriver'],
    'def':
        'a hand tool for driving screws; has a tip that fits into the head of a screw',
    'name':
        'screwdriver'
}, {
    'frequency': 'c',
    'id': 943,
    'synset': 'scrub_brush.n.01',
    'synonyms': ['scrubbing_brush'],
    'def': 'a brush with short stiff bristles for heavy cleaning',
    'name': 'scrubbing_brush'
}, {
    'frequency': 'c',
    'id': 944,
    'synset': 'sculpture.n.01',
    'synonyms': ['sculpture'],
    'def': 'a three-dimensional work of art',
    'name': 'sculpture'
}, {
    'frequency':
        'r',
    'id':
        945,
    'synset':
        'seabird.n.01',
    'synonyms': ['seabird', 'seafowl'],
    'def':
        'a bird that frequents coastal waters and the open ocean: gulls; pelicans; gannets; cormorants; albatrosses; petrels; etc.',
    'name':
        'seabird'
}, {
    'frequency':
        'r',
    'id':
        946,
    'synset':
        'seahorse.n.02',
    'synonyms': ['seahorse'],
    'def':
        'small fish with horse-like heads bent sharply downward and curled tails',
    'name':
        'seahorse'
}, {
    'frequency': 'r',
    'id': 947,
    'synset': 'seaplane.n.01',
    'synonyms': ['seaplane', 'hydroplane'],
    'def': 'an airplane that can land on or take off from water',
    'name': 'seaplane'
}, {
    'frequency': 'c',
    'id': 948,
    'synset': 'seashell.n.01',
    'synonyms': ['seashell'],
    'def': 'the shell of a marine organism',
    'name': 'seashell'
}, {
    'frequency': 'r',
    'id': 949,
    'synset': 'seedling.n.01',
    'synonyms': ['seedling'],
    'def': 'young plant or tree grown from a seed',
    'name': 'seedling'
}, {
    'frequency': 'c',
    'id': 950,
    'synset': 'serving_dish.n.01',
    'synonyms': ['serving_dish'],
    'def': 'a dish used for serving food',
    'name': 'serving_dish'
}, {
    'frequency': 'r',
    'id': 951,
    'synset': 'sewing_machine.n.01',
    'synonyms': ['sewing_machine'],
    'def': 'a textile machine used as a home appliance for sewing',
    'name': 'sewing_machine'
}, {
    'frequency': 'r',
    'id': 952,
    'synset': 'shaker.n.03',
    'synonyms': ['shaker'],
    'def': 'a container in which something can be shaken',
    'name': 'shaker'
}, {
    'frequency':
        'c',
    'id':
        953,
    'synset':
        'shampoo.n.01',
    'synonyms': ['shampoo'],
    'def':
        'cleansing agent consisting of soaps or detergents used for washing the hair',
    'name':
        'shampoo'
}, {
    'frequency': 'r',
    'id': 954,
    'synset': 'shark.n.01',
    'synonyms': ['shark'],
    'def': 'typically large carnivorous fishes with sharpe teeth',
    'name': 'shark'
}, {
    'frequency':
        'r',
    'id':
        955,
    'synset':
        'sharpener.n.01',
    'synonyms': ['sharpener'],
    'def':
        'any implement that is used to make something (an edge or a point) sharper',
    'name':
        'sharpener'
}, {
    'frequency': 'r',
    'id': 956,
    'synset': 'sharpie.n.03',
    'synonyms': ['Sharpie'],
    'def': 'a pen with indelible ink that will write on any surface',
    'name': 'Sharpie'
}, {
    'frequency': 'r',
    'id': 957,
    'synset': 'shaver.n.03',
    'synonyms': ['shaver_(electric)', 'electric_shaver', 'electric_razor'],
    'def': 'a razor powered by an electric motor',
    'name': 'shaver_(electric)'
}, {
    'frequency':
        'c',
    'id':
        958,
    'synset':
        'shaving_cream.n.01',
    'synonyms': ['shaving_cream', 'shaving_soap'],
    'def':
        'toiletry consisting that forms a rich lather for softening the beard before shaving',
    'name':
        'shaving_cream'
}, {
    'frequency':
        'r',
    'id':
        959,
    'synset':
        'shawl.n.01',
    'synonyms': ['shawl'],
    'def':
        'cloak consisting of an oblong piece of cloth used to cover the head and shoulders',
    'name':
        'shawl'
}, {
    'frequency': 'r',
    'id': 960,
    'synset': 'shears.n.01',
    'synonyms': ['shears'],
    'def': 'large scissors with strong blades',
    'name': 'shears'
}, {
    'frequency': 'f',
    'id': 961,
    'synset': 'sheep.n.01',
    'synonyms': ['sheep'],
    'def': 'woolly usually horned ruminant mammal related to the goat',
    'name': 'sheep'
}, {
    'frequency':
        'r',
    'id':
        962,
    'synset':
        'shepherd_dog.n.01',
    'synonyms': ['shepherd_dog', 'sheepdog'],
    'def':
        'any of various usually long-haired breeds of dog reared to herd and guard sheep',
    'name':
        'shepherd_dog'
}, {
    'frequency': 'r',
    'id': 963,
    'synset': 'sherbert.n.01',
    'synonyms': ['sherbert', 'sherbet'],
    'def': 'a frozen dessert made primarily of fruit juice and sugar',
    'name': 'sherbert'
}, {
    'frequency': 'r',
    'id': 964,
    'synset': 'shield.n.02',
    'synonyms': ['shield'],
    'def': 'armor carried on the arm to intercept blows',
    'name': 'shield'
}, {
    'frequency': 'f',
    'id': 965,
    'synset': 'shirt.n.01',
    'synonyms': ['shirt'],
    'def': 'a garment worn on the upper half of the body',
    'name': 'shirt'
}, {
    'frequency': 'f',
    'id': 966,
    'synset': 'shoe.n.01',
    'synonyms': ['shoe', 'sneaker_(type_of_shoe)', 'tennis_shoe'],
    'def': 'common footwear covering the foot',
    'name': 'shoe'
}, {
    'frequency':
        'c',
    'id':
        967,
    'synset':
        'shopping_bag.n.01',
    'synonyms': ['shopping_bag'],
    'def':
        'a bag made of plastic or strong paper (often with handles); used to transport goods after shopping',
    'name':
        'shopping_bag'
}, {
    'frequency': 'c',
    'id': 968,
    'synset': 'shopping_cart.n.01',
    'synonyms': ['shopping_cart'],
    'def': 'a handcart that holds groceries or other goods while shopping',
    'name': 'shopping_cart'
}, {
    'frequency': 'f',
    'id': 969,
    'synset': 'short_pants.n.01',
    'synonyms': ['short_pants', 'shorts_(clothing)', 'trunks_(clothing)'],
    'def': 'trousers that end at or above the knee',
    'name': 'short_pants'
}, {
    'frequency': 'r',
    'id': 970,
    'synset': 'shot_glass.n.01',
    'synonyms': ['shot_glass'],
    'def': 'a small glass adequate to hold a single swallow of whiskey',
    'name': 'shot_glass'
}, {
    'frequency':
        'c',
    'id':
        971,
    'synset':
        'shoulder_bag.n.01',
    'synonyms': ['shoulder_bag'],
    'def':
        'a large handbag that can be carried by a strap looped over the shoulder',
    'name':
        'shoulder_bag'
}, {
    'frequency': 'c',
    'id': 972,
    'synset': 'shovel.n.01',
    'synonyms': ['shovel'],
    'def': 'a hand tool for lifting loose material such as snow, dirt, etc.',
    'name': 'shovel'
}, {
    'frequency': 'f',
    'id': 973,
    'synset': 'shower.n.01',
    'synonyms': ['shower_head'],
    'def': 'a plumbing fixture that sprays water over you',
    'name': 'shower_head'
}, {
    'frequency': 'f',
    'id': 974,
    'synset': 'shower_curtain.n.01',
    'synonyms': ['shower_curtain'],
    'def': 'a curtain that keeps water from splashing out of the shower area',
    'name': 'shower_curtain'
}, {
    'frequency': 'r',
    'id': 975,
    'synset': 'shredder.n.01',
    'synonyms': ['shredder_(for_paper)'],
    'def': 'a device that shreds documents',
    'name': 'shredder_(for_paper)'
}, {
    'frequency':
        'r',
    'id':
        976,
    'synset':
        'sieve.n.01',
    'synonyms': ['sieve', 'screen_(sieve)'],
    'def':
        'a strainer for separating lumps from powdered material or grading particles',
    'name':
        'sieve'
}, {
    'frequency': 'f',
    'id': 977,
    'synset': 'signboard.n.01',
    'synonyms': ['signboard'],
    'def': 'structure displaying a board on which advertisements can be posted',
    'name': 'signboard'
}, {
    'frequency': 'c',
    'id': 978,
    'synset': 'silo.n.01',
    'synonyms': ['silo'],
    'def': 'a cylindrical tower used for storing goods',
    'name': 'silo'
}, {
    'frequency':
        'f',
    'id':
        979,
    'synset':
        'sink.n.01',
    'synonyms': ['sink'],
    'def':
        'plumbing fixture consisting of a water basin fixed to a wall or floor and having a drainpipe',
    'name':
        'sink'
}, {
    'frequency':
        'f',
    'id':
        980,
    'synset':
        'skateboard.n.01',
    'synonyms': ['skateboard'],
    'def':
        'a board with wheels that is ridden in a standing or crouching position and propelled by foot',
    'name':
        'skateboard'
}, {
    'frequency': 'c',
    'id': 981,
    'synset': 'skewer.n.01',
    'synonyms': ['skewer'],
    'def': 'a long pin for holding meat in position while it is being roasted',
    'name': 'skewer'
}, {
    'frequency': 'f',
    'id': 982,
    'synset': 'ski.n.01',
    'synonyms': ['ski'],
    'def': 'sports equipment for skiing on snow',
    'name': 'ski'
}, {
    'frequency': 'f',
    'id': 983,
    'synset': 'ski_boot.n.01',
    'synonyms': ['ski_boot'],
    'def': 'a stiff boot that is fastened to a ski with a ski binding',
    'name': 'ski_boot'
}, {
    'frequency': 'f',
    'id': 984,
    'synset': 'ski_parka.n.01',
    'synonyms': ['ski_parka', 'ski_jacket'],
    'def': 'a parka to be worn while skiing',
    'name': 'ski_parka'
}, {
    'frequency': 'f',
    'id': 985,
    'synset': 'ski_pole.n.01',
    'synonyms': ['ski_pole'],
    'def': 'a pole with metal points used as an aid in skiing',
    'name': 'ski_pole'
}, {
    'frequency': 'f',
    'id': 986,
    'synset': 'skirt.n.02',
    'synonyms': ['skirt'],
    'def': 'a garment hanging from the waist; worn mainly by girls and women',
    'name': 'skirt'
}, {
    'frequency':
        'c',
    'id':
        987,
    'synset':
        'sled.n.01',
    'synonyms': ['sled', 'sledge', 'sleigh'],
    'def':
        'a vehicle or flat object for transportation over snow by sliding or pulled by dogs, etc.',
    'name':
        'sled'
}, {
    'frequency': 'c',
    'id': 988,
    'synset': 'sleeping_bag.n.01',
    'synonyms': ['sleeping_bag'],
    'def': 'large padded bag designed to be slept in outdoors',
    'name': 'sleeping_bag'
}, {
    'frequency':
        'r',
    'id':
        989,
    'synset':
        'sling.n.05',
    'synonyms': ['sling_(bandage)', 'triangular_bandage'],
    'def':
        'bandage to support an injured forearm; slung over the shoulder or neck',
    'name':
        'sling_(bandage)'
}, {
    'frequency':
        'c',
    'id':
        990,
    'synset':
        'slipper.n.01',
    'synonyms': ['slipper_(footwear)', 'carpet_slipper_(footwear)'],
    'def':
        'low footwear that can be slipped on and off easily; usually worn indoors',
    'name':
        'slipper_(footwear)'
}, {
    'frequency':
        'r',
    'id':
        991,
    'synset':
        'smoothie.n.02',
    'synonyms': ['smoothie'],
    'def':
        'a thick smooth drink consisting of fresh fruit pureed with ice cream or yoghurt or milk',
    'name':
        'smoothie'
}, {
    'frequency': 'r',
    'id': 992,
    'synset': 'snake.n.01',
    'synonyms': ['snake', 'serpent'],
    'def': 'limbless scaly elongate reptile; some are venomous',
    'name': 'snake'
}, {
    'frequency':
        'f',
    'id':
        993,
    'synset':
        'snowboard.n.01',
    'synonyms': ['snowboard'],
    'def':
        'a board that resembles a broad ski or a small surfboard; used in a standing position to slide down snow-covered slopes',
    'name':
        'snowboard'
}, {
    'frequency': 'c',
    'id': 994,
    'synset': 'snowman.n.01',
    'synonyms': ['snowman'],
    'def': 'a figure of a person made of packed snow',
    'name': 'snowman'
}, {
    'frequency': 'c',
    'id': 995,
    'synset': 'snowmobile.n.01',
    'synonyms': ['snowmobile'],
    'def': 'tracked vehicle for travel on snow having skis in front',
    'name': 'snowmobile'
}, {
    'frequency': 'f',
    'id': 996,
    'synset': 'soap.n.01',
    'synonyms': ['soap'],
    'def': 'a cleansing agent made from the salts of vegetable or animal fats',
    'name': 'soap'
}, {
    'frequency':
        'f',
    'id':
        997,
    'synset':
        'soccer_ball.n.01',
    'synonyms': ['soccer_ball'],
    'def':
        "an inflated ball used in playing soccer (called `football' outside of the United States)",
    'name':
        'soccer_ball'
}, {
    'frequency':
        'f',
    'id':
        998,
    'synset':
        'sock.n.01',
    'synonyms': ['sock'],
    'def':
        'cloth covering for the foot; worn inside the shoe; reaches to between the ankle and the knee',
    'name':
        'sock'
}, {
    'frequency': 'r',
    'id': 999,
    'synset': 'soda_fountain.n.02',
    'synonyms': ['soda_fountain'],
    'def': 'an apparatus for dispensing soda water',
    'name': 'soda_fountain'
}, {
    'frequency': 'r',
    'id': 1000,
    'synset': 'soda_water.n.01',
    'synonyms': ['carbonated_water', 'club_soda', 'seltzer', 'sparkling_water'],
    'def': 'effervescent beverage artificially charged with carbon dioxide',
    'name': 'carbonated_water'
}, {
    'frequency': 'f',
    'id': 1001,
    'synset': 'sofa.n.01',
    'synonyms': ['sofa', 'couch', 'lounge'],
    'def': 'an upholstered seat for more than one person',
    'name': 'sofa'
}, {
    'frequency': 'r',
    'id': 1002,
    'synset': 'softball.n.01',
    'synonyms': ['softball'],
    'def': 'ball used in playing softball',
    'name': 'softball'
}, {
    'frequency':
        'c',
    'id':
        1003,
    'synset':
        'solar_array.n.01',
    'synonyms': ['solar_array', 'solar_battery', 'solar_panel'],
    'def':
        'electrical device consisting of a large array of connected solar cells',
    'name':
        'solar_array'
}, {
    'frequency':
        'r',
    'id':
        1004,
    'synset':
        'sombrero.n.02',
    'synonyms': ['sombrero'],
    'def':
        'a straw hat with a tall crown and broad brim; worn in American southwest and in Mexico',
    'name':
        'sombrero'
}, {
    'frequency':
        'c',
    'id':
        1005,
    'synset':
        'soup.n.01',
    'synonyms': ['soup'],
    'def':
        'liquid food especially of meat or fish or vegetable stock often containing pieces of solid food',
    'name':
        'soup'
}, {
    'frequency': 'r',
    'id': 1006,
    'synset': 'soup_bowl.n.01',
    'synonyms': ['soup_bowl'],
    'def': 'a bowl for serving soup',
    'name': 'soup_bowl'
}, {
    'frequency': 'c',
    'id': 1007,
    'synset': 'soupspoon.n.01',
    'synonyms': ['soupspoon'],
    'def': 'a spoon with a rounded bowl for eating soup',
    'name': 'soupspoon'
}, {
    'frequency': 'c',
    'id': 1008,
    'synset': 'sour_cream.n.01',
    'synonyms': ['sour_cream', 'soured_cream'],
    'def': 'soured light cream',
    'name': 'sour_cream'
}, {
    'frequency':
        'r',
    'id':
        1009,
    'synset':
        'soya_milk.n.01',
    'synonyms': ['soya_milk', 'soybean_milk', 'soymilk'],
    'def':
        'a milk substitute containing soybean flour and water; used in some infant formulas and in making tofu',
    'name':
        'soya_milk'
}, {
    'frequency':
        'r',
    'id':
        1010,
    'synset':
        'space_shuttle.n.01',
    'synonyms': ['space_shuttle'],
    'def':
        "a reusable spacecraft with wings for a controlled descent through the Earth's atmosphere",
    'name':
        'space_shuttle'
}, {
    'frequency': 'r',
    'id': 1011,
    'synset': 'sparkler.n.02',
    'synonyms': ['sparkler_(fireworks)'],
    'def': 'a firework that burns slowly and throws out a shower of sparks',
    'name': 'sparkler_(fireworks)'
}, {
    'frequency':
        'f',
    'id':
        1012,
    'synset':
        'spatula.n.02',
    'synonyms': ['spatula'],
    'def':
        'a hand tool with a thin flexible blade used to mix or spread soft substances',
    'name':
        'spatula'
}, {
    'frequency': 'r',
    'id': 1013,
    'synset': 'spear.n.01',
    'synonyms': ['spear', 'lance'],
    'def': 'a long pointed rod used as a tool or weapon',
    'name': 'spear'
}, {
    'frequency':
        'f',
    'id':
        1014,
    'synset':
        'spectacles.n.01',
    'synonyms': ['spectacles', 'specs', 'eyeglasses', 'glasses'],
    'def':
        'optical instrument consisting of a frame that holds a pair of lenses for correcting defective vision',
    'name':
        'spectacles'
}, {
    'frequency': 'c',
    'id': 1015,
    'synset': 'spice_rack.n.01',
    'synonyms': ['spice_rack'],
    'def': 'a rack for displaying containers filled with spices',
    'name': 'spice_rack'
}, {
    'frequency':
        'r',
    'id':
        1016,
    'synset':
        'spider.n.01',
    'synonyms': ['spider'],
    'def':
        'predatory arachnid with eight legs, two poison fangs, two feelers, and usually two silk-spinning organs at the back end of the body',
    'name':
        'spider'
}, {
    'frequency': 'c',
    'id': 1017,
    'synset': 'sponge.n.01',
    'synonyms': ['sponge'],
    'def': 'a porous mass usable to absorb water typically used for cleaning',
    'name': 'sponge'
}, {
    'frequency':
        'f',
    'id':
        1018,
    'synset':
        'spoon.n.01',
    'synonyms': ['spoon'],
    'def':
        'a piece of cutlery with a shallow bowl-shaped container and a handle',
    'name':
        'spoon'
}, {
    'frequency': 'c',
    'id': 1019,
    'synset': 'sportswear.n.01',
    'synonyms': ['sportswear', 'athletic_wear', 'activewear'],
    'def': 'attire worn for sport or for casual wear',
    'name': 'sportswear'
}, {
    'frequency':
        'c',
    'id':
        1020,
    'synset':
        'spotlight.n.02',
    'synonyms': ['spotlight'],
    'def':
        'a lamp that produces a strong beam of light to illuminate a restricted area; used to focus attention of a stage performer',
    'name':
        'spotlight'
}, {
    'frequency': 'r',
    'id': 1021,
    'synset': 'squirrel.n.01',
    'synonyms': ['squirrel'],
    'def': 'a kind of arboreal rodent having a long bushy tail',
    'name': 'squirrel'
}, {
    'frequency':
        'c',
    'id':
        1022,
    'synset':
        'stapler.n.01',
    'synonyms': ['stapler_(stapling_machine)'],
    'def':
        'a machine that inserts staples into sheets of paper in order to fasten them together',
    'name':
        'stapler_(stapling_machine)'
}, {
    'frequency':
        'r',
    'id':
        1023,
    'synset':
        'starfish.n.01',
    'synonyms': ['starfish', 'sea_star'],
    'def':
        'echinoderms characterized by five arms extending from a central disk',
    'name':
        'starfish'
}, {
    'frequency': 'f',
    'id': 1024,
    'synset': 'statue.n.01',
    'synonyms': ['statue_(sculpture)'],
    'def': 'a sculpture representing a human or animal',
    'name': 'statue_(sculpture)'
}, {
    'frequency':
        'c',
    'id':
        1025,
    'synset':
        'steak.n.01',
    'synonyms': ['steak_(food)'],
    'def':
        'a slice of meat cut from the fleshy part of an animal or large fish',
    'name':
        'steak_(food)'
}, {
    'frequency': 'r',
    'id': 1026,
    'synset': 'steak_knife.n.01',
    'synonyms': ['steak_knife'],
    'def': 'a sharp table knife used in eating steak',
    'name': 'steak_knife'
}, {
    'frequency': 'r',
    'id': 1027,
    'synset': 'steamer.n.02',
    'synonyms': ['steamer_(kitchen_appliance)'],
    'def': 'a cooking utensil that can be used to cook food by steaming it',
    'name': 'steamer_(kitchen_appliance)'
}, {
    'frequency': 'f',
    'id': 1028,
    'synset': 'steering_wheel.n.01',
    'synonyms': ['steering_wheel'],
    'def': 'a handwheel that is used for steering',
    'name': 'steering_wheel'
}, {
    'frequency':
        'r',
    'id':
        1029,
    'synset':
        'stencil.n.01',
    'synonyms': ['stencil'],
    'def':
        'a sheet of material (metal, plastic, etc.) that has been perforated with a pattern; ink or paint can pass through the perforations to create the printed pattern on the surface below',
    'name':
        'stencil'
}, {
    'frequency': 'r',
    'id': 1030,
    'synset': 'step_ladder.n.01',
    'synonyms': ['stepladder'],
    'def': 'a folding portable ladder hinged at the top',
    'name': 'stepladder'
}, {
    'frequency': 'c',
    'id': 1031,
    'synset': 'step_stool.n.01',
    'synonyms': ['step_stool'],
    'def': 'a stool that has one or two steps that fold under the seat',
    'name': 'step_stool'
}, {
    'frequency': 'c',
    'id': 1032,
    'synset': 'stereo.n.01',
    'synonyms': ['stereo_(sound_system)'],
    'def': 'electronic device for playing audio',
    'name': 'stereo_(sound_system)'
}, {
    'frequency': 'r',
    'id': 1033,
    'synset': 'stew.n.02',
    'synonyms': ['stew'],
    'def': 'food prepared by stewing especially meat or fish with vegetables',
    'name': 'stew'
}, {
    'frequency': 'r',
    'id': 1034,
    'synset': 'stirrer.n.02',
    'synonyms': ['stirrer'],
    'def': 'an implement used for stirring',
    'name': 'stirrer'
}, {
    'frequency': 'f',
    'id': 1035,
    'synset': 'stirrup.n.01',
    'synonyms': ['stirrup'],
    'def': "support consisting of metal loops into which rider's feet go",
    'name': 'stirrup'
}, {
    'frequency':
        'c',
    'id':
        1036,
    'synset':
        'stocking.n.01',
    'synonyms': ['stockings_(leg_wear)'],
    'def':
        'close-fitting hosiery to cover the foot and leg; come in matched pairs',
    'name':
        'stockings_(leg_wear)'
}, {
    'frequency': 'f',
    'id': 1037,
    'synset': 'stool.n.01',
    'synonyms': ['stool'],
    'def': 'a simple seat without a back or arms',
    'name': 'stool'
}, {
    'frequency':
        'f',
    'id':
        1038,
    'synset':
        'stop_sign.n.01',
    'synonyms': ['stop_sign'],
    'def':
        'a traffic sign to notify drivers that they must come to a complete stop',
    'name':
        'stop_sign'
}, {
    'frequency':
        'f',
    'id':
        1039,
    'synset':
        'stoplight.n.01',
    'synonyms': ['brake_light'],
    'def':
        'a red light on the rear of a motor vehicle that signals when the brakes are applied',
    'name':
        'brake_light'
}, {
    'frequency': 'f',
    'id': 1040,
    'synset': 'stove.n.01',
    'synonyms': [
        'stove', 'kitchen_stove', 'range_(kitchen_appliance)', 'kitchen_range',
        'cooking_stove'
    ],
    'def': 'a kitchen appliance used for cooking food',
    'name': 'stove'
}, {
    'frequency':
        'c',
    'id':
        1041,
    'synset':
        'strainer.n.01',
    'synonyms': ['strainer'],
    'def':
        'a filter to retain larger pieces while smaller pieces and liquids pass through',
    'name':
        'strainer'
}, {
    'frequency':
        'f',
    'id':
        1042,
    'synset':
        'strap.n.01',
    'synonyms': ['strap'],
    'def':
        'an elongated strip of material for binding things together or holding',
    'name':
        'strap'
}, {
    'frequency': 'f',
    'id': 1043,
    'synset': 'straw.n.04',
    'synonyms': ['straw_(for_drinking)', 'drinking_straw'],
    'def': 'a thin paper or plastic tube used to suck liquids into the mouth',
    'name': 'straw_(for_drinking)'
}, {
    'frequency': 'f',
    'id': 1044,
    'synset': 'strawberry.n.01',
    'synonyms': ['strawberry'],
    'def': 'sweet fleshy red fruit',
    'name': 'strawberry'
}, {
    'frequency': 'f',
    'id': 1045,
    'synset': 'street_sign.n.01',
    'synonyms': ['street_sign'],
    'def': 'a sign visible from the street',
    'name': 'street_sign'
}, {
    'frequency': 'f',
    'id': 1046,
    'synset': 'streetlight.n.01',
    'synonyms': ['streetlight', 'street_lamp'],
    'def': 'a lamp supported on a lamppost; for illuminating a street',
    'name': 'streetlight'
}, {
    'frequency': 'r',
    'id': 1047,
    'synset': 'string_cheese.n.01',
    'synonyms': ['string_cheese'],
    'def': 'cheese formed in long strings twisted together',
    'name': 'string_cheese'
}, {
    'frequency': 'r',
    'id': 1048,
    'synset': 'stylus.n.02',
    'synonyms': ['stylus'],
    'def': 'a pointed tool for writing or drawing or engraving',
    'name': 'stylus'
}, {
    'frequency':
        'r',
    'id':
        1049,
    'synset':
        'subwoofer.n.01',
    'synonyms': ['subwoofer'],
    'def':
        'a loudspeaker that is designed to reproduce very low bass frequencies',
    'name':
        'subwoofer'
}, {
    'frequency': 'r',
    'id': 1050,
    'synset': 'sugar_bowl.n.01',
    'synonyms': ['sugar_bowl'],
    'def': 'a dish in which sugar is served',
    'name': 'sugar_bowl'
}, {
    'frequency':
        'r',
    'id':
        1051,
    'synset':
        'sugarcane.n.01',
    'synonyms': ['sugarcane_(plant)'],
    'def':
        'juicy canes whose sap is a source of molasses and commercial sugar; fresh canes are sometimes chewed for the juice',
    'name':
        'sugarcane_(plant)'
}, {
    'frequency':
        'c',
    'id':
        1052,
    'synset':
        'suit.n.01',
    'synonyms': ['suit_(clothing)'],
    'def':
        'a set of garments (usually including a jacket and trousers or skirt) for outerwear all of the same fabric and color',
    'name':
        'suit_(clothing)'
}, {
    'frequency':
        'c',
    'id':
        1053,
    'synset':
        'sunflower.n.01',
    'synonyms': ['sunflower'],
    'def':
        'any plant of the genus Helianthus having large flower heads with dark disk florets and showy yellow rays',
    'name':
        'sunflower'
}, {
    'frequency':
        'f',
    'id':
        1054,
    'synset':
        'sunglasses.n.01',
    'synonyms': ['sunglasses'],
    'def':
        'spectacles that are darkened or polarized to protect the eyes from the glare of the sun',
    'name':
        'sunglasses'
}, {
    'frequency':
        'c',
    'id':
        1055,
    'synset':
        'sunhat.n.01',
    'synonyms': ['sunhat'],
    'def':
        'a hat with a broad brim that protects the face from direct exposure to the sun',
    'name':
        'sunhat'
}, {
    'frequency':
        'r',
    'id':
        1056,
    'synset':
        'sunscreen.n.01',
    'synonyms': ['sunscreen', 'sunblock'],
    'def':
        'a cream spread on the skin; contains a chemical to filter out ultraviolet light and so protect from sunburn',
    'name':
        'sunscreen'
}, {
    'frequency': 'f',
    'id': 1057,
    'synset': 'surfboard.n.01',
    'synonyms': ['surfboard'],
    'def': 'a narrow buoyant board for riding surf',
    'name': 'surfboard'
}, {
    'frequency': 'c',
    'id': 1058,
    'synset': 'sushi.n.01',
    'synonyms': ['sushi'],
    'def': 'rice (with raw fish) wrapped in seaweed',
    'name': 'sushi'
}, {
    'frequency':
        'c',
    'id':
        1059,
    'synset':
        'swab.n.02',
    'synonyms': ['mop'],
    'def':
        'cleaning implement consisting of absorbent material fastened to a handle; for cleaning floors',
    'name':
        'mop'
}, {
    'frequency': 'c',
    'id': 1060,
    'synset': 'sweat_pants.n.01',
    'synonyms': ['sweat_pants'],
    'def': 'loose-fitting trousers with elastic cuffs; worn by athletes',
    'name': 'sweat_pants'
}, {
    'frequency':
        'c',
    'id':
        1061,
    'synset':
        'sweatband.n.02',
    'synonyms': ['sweatband'],
    'def':
        'a band of material tied around the forehead or wrist to absorb sweat',
    'name':
        'sweatband'
}, {
    'frequency': 'f',
    'id': 1062,
    'synset': 'sweater.n.01',
    'synonyms': ['sweater'],
    'def': 'a crocheted or knitted garment covering the upper part of the body',
    'name': 'sweater'
}, {
    'frequency':
        'f',
    'id':
        1063,
    'synset':
        'sweatshirt.n.01',
    'synonyms': ['sweatshirt'],
    'def':
        'cotton knit pullover with long sleeves worn during athletic activity',
    'name':
        'sweatshirt'
}, {
    'frequency': 'c',
    'id': 1064,
    'synset': 'sweet_potato.n.02',
    'synonyms': ['sweet_potato'],
    'def': 'the edible tuberous root of the sweet potato vine',
    'name': 'sweet_potato'
}, {
    'frequency': 'f',
    'id': 1065,
    'synset': 'swimsuit.n.01',
    'synonyms': [
        'swimsuit', 'swimwear', 'bathing_suit', 'swimming_costume',
        'bathing_costume', 'swimming_trunks', 'bathing_trunks'
    ],
    'def': 'garment worn for swimming',
    'name': 'swimsuit'
}, {
    'frequency': 'c',
    'id': 1066,
    'synset': 'sword.n.01',
    'synonyms': ['sword'],
    'def': 'a cutting or thrusting weapon that has a long metal blade',
    'name': 'sword'
}, {
    'frequency': 'r',
    'id': 1067,
    'synset': 'syringe.n.01',
    'synonyms': ['syringe'],
    'def': 'a medical instrument used to inject or withdraw fluids',
    'name': 'syringe'
}, {
    'frequency':
        'r',
    'id':
        1068,
    'synset':
        'tabasco.n.02',
    'synonyms': ['Tabasco_sauce'],
    'def':
        'very spicy sauce (trade name Tabasco) made from fully-aged red peppers',
    'name':
        'Tabasco_sauce'
}, {
    'frequency': 'r',
    'id': 1069,
    'synset': 'table-tennis_table.n.01',
    'synonyms': ['table-tennis_table', 'ping-pong_table'],
    'def': 'a table used for playing table tennis',
    'name': 'table-tennis_table'
}, {
    'frequency':
        'f',
    'id':
        1070,
    'synset':
        'table.n.02',
    'synonyms': ['table'],
    'def':
        'a piece of furniture having a smooth flat top that is usually supported by one or more vertical legs',
    'name':
        'table'
}, {
    'frequency': 'c',
    'id': 1071,
    'synset': 'table_lamp.n.01',
    'synonyms': ['table_lamp'],
    'def': 'a lamp that sits on a table',
    'name': 'table_lamp'
}, {
    'frequency': 'f',
    'id': 1072,
    'synset': 'tablecloth.n.01',
    'synonyms': ['tablecloth'],
    'def': 'a covering spread over a dining table',
    'name': 'tablecloth'
}, {
    'frequency': 'r',
    'id': 1073,
    'synset': 'tachometer.n.01',
    'synonyms': ['tachometer'],
    'def': 'measuring instrument for indicating speed of rotation',
    'name': 'tachometer'
}, {
    'frequency': 'r',
    'id': 1074,
    'synset': 'taco.n.02',
    'synonyms': ['taco'],
    'def': 'a small tortilla cupped around a filling',
    'name': 'taco'
}, {
    'frequency':
        'f',
    'id':
        1075,
    'synset':
        'tag.n.02',
    'synonyms': ['tag'],
    'def':
        'a label associated with something for the purpose of identification or information',
    'name':
        'tag'
}, {
    'frequency': 'f',
    'id': 1076,
    'synset': 'taillight.n.01',
    'synonyms': ['taillight', 'rear_light'],
    'def': 'lamp (usually red) mounted at the rear of a motor vehicle',
    'name': 'taillight'
}, {
    'frequency':
        'r',
    'id':
        1077,
    'synset':
        'tambourine.n.01',
    'synonyms': ['tambourine'],
    'def':
        'a shallow drum with a single drumhead and with metallic disks in the sides',
    'name':
        'tambourine'
}, {
    'frequency':
        'r',
    'id':
        1078,
    'synset':
        'tank.n.01',
    'synonyms': [
        'army_tank', 'armored_combat_vehicle', 'armoured_combat_vehicle'
    ],
    'def':
        'an enclosed armored military vehicle; has a cannon and moves on caterpillar treads',
    'name':
        'army_tank'
}, {
    'frequency': 'c',
    'id': 1079,
    'synset': 'tank.n.02',
    'synonyms': ['tank_(storage_vessel)', 'storage_tank'],
    'def': 'a large (usually metallic) vessel for holding gases or liquids',
    'name': 'tank_(storage_vessel)'
}, {
    'frequency':
        'f',
    'id':
        1080,
    'synset':
        'tank_top.n.01',
    'synonyms': ['tank_top_(clothing)'],
    'def':
        'a tight-fitting sleeveless shirt with wide shoulder straps and low neck and no front opening',
    'name':
        'tank_top_(clothing)'
}, {
    'frequency':
        'c',
    'id':
        1081,
    'synset':
        'tape.n.01',
    'synonyms': ['tape_(sticky_cloth_or_paper)'],
    'def':
        'a long thin piece of cloth or paper as used for binding or fastening',
    'name':
        'tape_(sticky_cloth_or_paper)'
}, {
    'frequency':
        'c',
    'id':
        1082,
    'synset':
        'tape.n.04',
    'synonyms': ['tape_measure', 'measuring_tape'],
    'def':
        'measuring instrument consisting of a narrow strip (cloth or metal) marked in inches or centimeters and used for measuring lengths',
    'name':
        'tape_measure'
}, {
    'frequency':
        'c',
    'id':
        1083,
    'synset':
        'tapestry.n.02',
    'synonyms': ['tapestry'],
    'def':
        'a heavy textile with a woven design; used for curtains and upholstery',
    'name':
        'tapestry'
}, {
    'frequency': 'f',
    'id': 1084,
    'synset': 'tarpaulin.n.01',
    'synonyms': ['tarp'],
    'def': 'waterproofed canvas',
    'name': 'tarp'
}, {
    'frequency': 'c',
    'id': 1085,
    'synset': 'tartan.n.01',
    'synonyms': ['tartan', 'plaid'],
    'def': 'a cloth having a crisscross design',
    'name': 'tartan'
}, {
    'frequency': 'c',
    'id': 1086,
    'synset': 'tassel.n.01',
    'synonyms': ['tassel'],
    'def': 'adornment consisting of a bunch of cords fastened at one end',
    'name': 'tassel'
}, {
    'frequency': 'r',
    'id': 1087,
    'synset': 'tea_bag.n.01',
    'synonyms': ['tea_bag'],
    'def': 'a measured amount of tea in a bag for an individual serving of tea',
    'name': 'tea_bag'
}, {
    'frequency': 'c',
    'id': 1088,
    'synset': 'teacup.n.02',
    'synonyms': ['teacup'],
    'def': 'a cup from which tea is drunk',
    'name': 'teacup'
}, {
    'frequency': 'c',
    'id': 1089,
    'synset': 'teakettle.n.01',
    'synonyms': ['teakettle'],
    'def': 'kettle for boiling water to make tea',
    'name': 'teakettle'
}, {
    'frequency': 'c',
    'id': 1090,
    'synset': 'teapot.n.01',
    'synonyms': ['teapot'],
    'def': 'pot for brewing tea; usually has a spout and handle',
    'name': 'teapot'
}, {
    'frequency':
        'f',
    'id':
        1091,
    'synset':
        'teddy.n.01',
    'synonyms': ['teddy_bear'],
    'def':
        "plaything consisting of a child's toy bear (usually plush and stuffed with soft materials)",
    'name':
        'teddy_bear'
}, {
    'frequency': 'f',
    'id': 1092,
    'synset': 'telephone.n.01',
    'synonyms': ['telephone', 'phone', 'telephone_set'],
    'def': 'electronic device for communicating by voice over long distances',
    'name': 'telephone'
}, {
    'frequency': 'c',
    'id': 1093,
    'synset': 'telephone_booth.n.01',
    'synonyms': [
        'telephone_booth', 'phone_booth', 'call_box', 'telephone_box',
        'telephone_kiosk'
    ],
    'def': 'booth for using a telephone',
    'name': 'telephone_booth'
}, {
    'frequency': 'f',
    'id': 1094,
    'synset': 'telephone_pole.n.01',
    'synonyms': ['telephone_pole', 'telegraph_pole', 'telegraph_post'],
    'def': 'tall pole supporting telephone wires',
    'name': 'telephone_pole'
}, {
    'frequency': 'r',
    'id': 1095,
    'synset': 'telephoto_lens.n.01',
    'synonyms': ['telephoto_lens', 'zoom_lens'],
    'def': 'a camera lens that magnifies the image',
    'name': 'telephoto_lens'
}, {
    'frequency': 'c',
    'id': 1096,
    'synset': 'television_camera.n.01',
    'synonyms': ['television_camera', 'tv_camera'],
    'def': 'television equipment for capturing and recording video',
    'name': 'television_camera'
}, {
    'frequency':
        'f',
    'id':
        1097,
    'synset':
        'television_receiver.n.01',
    'synonyms': ['television_set', 'tv', 'tv_set'],
    'def':
        'an electronic device that receives television signals and displays them on a screen',
    'name':
        'television_set'
}, {
    'frequency': 'f',
    'id': 1098,
    'synset': 'tennis_ball.n.01',
    'synonyms': ['tennis_ball'],
    'def': 'ball about the size of a fist used in playing tennis',
    'name': 'tennis_ball'
}, {
    'frequency': 'f',
    'id': 1099,
    'synset': 'tennis_racket.n.01',
    'synonyms': ['tennis_racket'],
    'def': 'a racket used to play tennis',
    'name': 'tennis_racket'
}, {
    'frequency': 'r',
    'id': 1100,
    'synset': 'tequila.n.01',
    'synonyms': ['tequila'],
    'def': 'Mexican liquor made from fermented juices of an agave plant',
    'name': 'tequila'
}, {
    'frequency': 'c',
    'id': 1101,
    'synset': 'thermometer.n.01',
    'synonyms': ['thermometer'],
    'def': 'measuring instrument for measuring temperature',
    'name': 'thermometer'
}, {
    'frequency': 'c',
    'id': 1102,
    'synset': 'thermos.n.01',
    'synonyms': ['thermos_bottle'],
    'def': 'vacuum flask that preserves temperature of hot or cold drinks',
    'name': 'thermos_bottle'
}, {
    'frequency':
        'c',
    'id':
        1103,
    'synset':
        'thermostat.n.01',
    'synonyms': ['thermostat'],
    'def':
        'a regulator for automatically regulating temperature by starting or stopping the supply of heat',
    'name':
        'thermostat'
}, {
    'frequency':
        'r',
    'id':
        1104,
    'synset':
        'thimble.n.02',
    'synonyms': ['thimble'],
    'def':
        'a small metal cap to protect the finger while sewing; can be used as a small container',
    'name':
        'thimble'
}, {
    'frequency':
        'c',
    'id':
        1105,
    'synset':
        'thread.n.01',
    'synonyms': ['thread', 'yarn'],
    'def':
        'a fine cord of twisted fibers (of cotton or silk or wool or nylon etc.) used in sewing and weaving',
    'name':
        'thread'
}, {
    'frequency': 'c',
    'id': 1106,
    'synset': 'thumbtack.n.01',
    'synonyms': ['thumbtack', 'drawing_pin', 'pushpin'],
    'def': 'a tack for attaching papers to a bulletin board or drawing board',
    'name': 'thumbtack'
}, {
    'frequency': 'c',
    'id': 1107,
    'synset': 'tiara.n.01',
    'synonyms': ['tiara'],
    'def': 'a jeweled headdress worn by women on formal occasions',
    'name': 'tiara'
}, {
    'frequency':
        'c',
    'id':
        1108,
    'synset':
        'tiger.n.02',
    'synonyms': ['tiger'],
    'def':
        'large feline of forests in most of Asia having a tawny coat with black stripes',
    'name':
        'tiger'
}, {
    'frequency':
        'c',
    'id':
        1109,
    'synset':
        'tights.n.01',
    'synonyms': ['tights_(clothing)', 'leotards'],
    'def':
        'skintight knit hose covering the body from the waist to the feet worn by acrobats and dancers and as stockings by women and girls',
    'name':
        'tights_(clothing)'
}, {
    'frequency': 'c',
    'id': 1110,
    'synset': 'timer.n.01',
    'synonyms': ['timer', 'stopwatch'],
    'def': 'a timepiece that measures a time interval and signals its end',
    'name': 'timer'
}, {
    'frequency': 'f',
    'id': 1111,
    'synset': 'tinfoil.n.01',
    'synonyms': ['tinfoil'],
    'def': 'foil made of tin or an alloy of tin and lead',
    'name': 'tinfoil'
}, {
    'frequency': 'r',
    'id': 1112,
    'synset': 'tinsel.n.01',
    'synonyms': ['tinsel'],
    'def': 'a showy decoration that is basically valueless',
    'name': 'tinsel'
}, {
    'frequency': 'f',
    'id': 1113,
    'synset': 'tissue.n.02',
    'synonyms': ['tissue_paper'],
    'def': 'a soft thin (usually translucent) paper',
    'name': 'tissue_paper'
}, {
    'frequency': 'c',
    'id': 1114,
    'synset': 'toast.n.01',
    'synonyms': ['toast_(food)'],
    'def': 'slice of bread that has been toasted',
    'name': 'toast_(food)'
}, {
    'frequency': 'f',
    'id': 1115,
    'synset': 'toaster.n.02',
    'synonyms': ['toaster'],
    'def': 'a kitchen appliance (usually electric) for toasting bread',
    'name': 'toaster'
}, {
    'frequency':
        'c',
    'id':
        1116,
    'synset':
        'toaster_oven.n.01',
    'synonyms': ['toaster_oven'],
    'def':
        'kitchen appliance consisting of a small electric oven for toasting or warming food',
    'name':
        'toaster_oven'
}, {
    'frequency': 'f',
    'id': 1117,
    'synset': 'toilet.n.02',
    'synonyms': ['toilet'],
    'def': 'a plumbing fixture for defecation and urination',
    'name': 'toilet'
}, {
    'frequency': 'f',
    'id': 1118,
    'synset': 'toilet_tissue.n.01',
    'synonyms': ['toilet_tissue', 'toilet_paper', 'bathroom_tissue'],
    'def': 'a soft thin absorbent paper for use in toilets',
    'name': 'toilet_tissue'
}, {
    'frequency': 'f',
    'id': 1119,
    'synset': 'tomato.n.01',
    'synonyms': ['tomato'],
    'def': 'mildly acid red or yellow pulpy fruit eaten as a vegetable',
    'name': 'tomato'
}, {
    'frequency':
        'c',
    'id':
        1120,
    'synset':
        'tongs.n.01',
    'synonyms': ['tongs'],
    'def':
        'any of various devices for taking hold of objects; usually have two hinged legs with handles above and pointed hooks below',
    'name':
        'tongs'
}, {
    'frequency': 'c',
    'id': 1121,
    'synset': 'toolbox.n.01',
    'synonyms': ['toolbox'],
    'def': 'a box or chest or cabinet for holding hand tools',
    'name': 'toolbox'
}, {
    'frequency': 'f',
    'id': 1122,
    'synset': 'toothbrush.n.01',
    'synonyms': ['toothbrush'],
    'def': 'small brush; has long handle; used to clean teeth',
    'name': 'toothbrush'
}, {
    'frequency': 'f',
    'id': 1123,
    'synset': 'toothpaste.n.01',
    'synonyms': ['toothpaste'],
    'def': 'a dentifrice in the form of a paste',
    'name': 'toothpaste'
}, {
    'frequency':
        'c',
    'id':
        1124,
    'synset':
        'toothpick.n.01',
    'synonyms': ['toothpick'],
    'def':
        'pick consisting of a small strip of wood or plastic; used to pick food from between the teeth',
    'name':
        'toothpick'
}, {
    'frequency': 'c',
    'id': 1125,
    'synset': 'top.n.09',
    'synonyms': ['cover'],
    'def': 'covering for a hole (especially a hole in the top of a container)',
    'name': 'cover'
}, {
    'frequency': 'c',
    'id': 1126,
    'synset': 'tortilla.n.01',
    'synonyms': ['tortilla'],
    'def': 'thin unleavened pancake made from cornmeal or wheat flour',
    'name': 'tortilla'
}, {
    'frequency':
        'c',
    'id':
        1127,
    'synset':
        'tow_truck.n.01',
    'synonyms': ['tow_truck'],
    'def':
        'a truck equipped to hoist and pull wrecked cars (or to remove cars from no-parking zones)',
    'name':
        'tow_truck'
}, {
    'frequency':
        'f',
    'id':
        1128,
    'synset':
        'towel.n.01',
    'synonyms': ['towel'],
    'def':
        'a rectangular piece of absorbent cloth (or paper) for drying or wiping',
    'name':
        'towel'
}, {
    'frequency': 'f',
    'id': 1129,
    'synset': 'towel_rack.n.01',
    'synonyms': ['towel_rack', 'towel_rail', 'towel_bar'],
    'def': 'a rack consisting of one or more bars on which towels can be hung',
    'name': 'towel_rack'
}, {
    'frequency': 'f',
    'id': 1130,
    'synset': 'toy.n.03',
    'synonyms': ['toy'],
    'def': 'a device regarded as providing amusement',
    'name': 'toy'
}, {
    'frequency':
        'c',
    'id':
        1131,
    'synset':
        'tractor.n.01',
    'synonyms': ['tractor_(farm_equipment)'],
    'def':
        'a wheeled vehicle with large wheels; used in farming and other applications',
    'name':
        'tractor_(farm_equipment)'
}, {
    'frequency':
        'f',
    'id':
        1132,
    'synset':
        'traffic_light.n.01',
    'synonyms': ['traffic_light'],
    'def':
        'a device to control vehicle traffic often consisting of three or more lights',
    'name':
        'traffic_light'
}, {
    'frequency':
        'r',
    'id':
        1133,
    'synset':
        'trail_bike.n.01',
    'synonyms': ['dirt_bike'],
    'def':
        'a lightweight motorcycle equipped with rugged tires and suspension for off-road use',
    'name':
        'dirt_bike'
}, {
    'frequency': 'c',
    'id': 1134,
    'synset': 'trailer_truck.n.01',
    'synonyms': [
        'trailer_truck', 'tractor_trailer', 'trucking_rig', 'articulated_lorry',
        'semi_truck'
    ],
    'def': 'a truck consisting of a tractor and trailer together',
    'name': 'trailer_truck'
}, {
    'frequency':
        'f',
    'id':
        1135,
    'synset':
        'train.n.01',
    'synonyms': ['train_(railroad_vehicle)', 'railroad_train'],
    'def':
        'public or private transport provided by a line of railway cars coupled together and drawn by a locomotive',
    'name':
        'train_(railroad_vehicle)'
}, {
    'frequency':
        'r',
    'id':
        1136,
    'synset':
        'trampoline.n.01',
    'synonyms': ['trampoline'],
    'def':
        'gymnastic apparatus consisting of a strong canvas sheet attached with springs to a metal frame',
    'name':
        'trampoline'
}, {
    'frequency':
        'f',
    'id':
        1137,
    'synset':
        'tray.n.01',
    'synonyms': ['tray'],
    'def':
        'an open receptacle for holding or displaying or serving articles or food',
    'name':
        'tray'
}, {
    'frequency': 'r',
    'id': 1138,
    'synset': 'tree_house.n.01',
    'synonyms': ['tree_house'],
    'def': '(NOT A TREE) a PLAYHOUSE built in the branches of a tree',
    'name': 'tree_house'
}, {
    'frequency': 'r',
    'id': 1139,
    'synset': 'trench_coat.n.01',
    'synonyms': ['trench_coat'],
    'def': 'a military style raincoat; belted with deep pockets',
    'name': 'trench_coat'
}, {
    'frequency':
        'r',
    'id':
        1140,
    'synset':
        'triangle.n.05',
    'synonyms': ['triangle_(musical_instrument)'],
    'def':
        'a percussion instrument consisting of a metal bar bent in the shape of an open triangle',
    'name':
        'triangle_(musical_instrument)'
}, {
    'frequency': 'r',
    'id': 1141,
    'synset': 'tricycle.n.01',
    'synonyms': ['tricycle'],
    'def': 'a vehicle with three wheels that is moved by foot pedals',
    'name': 'tricycle'
}, {
    'frequency': 'c',
    'id': 1142,
    'synset': 'tripod.n.01',
    'synonyms': ['tripod'],
    'def': 'a three-legged rack used for support',
    'name': 'tripod'
}, {
    'frequency':
        'f',
    'id':
        1143,
    'synset':
        'trouser.n.01',
    'synonyms': ['trousers', 'pants_(clothing)'],
    'def':
        'a garment extending from the waist to the knee or ankle, covering each leg separately',
    'name':
        'trousers'
}, {
    'frequency': 'f',
    'id': 1144,
    'synset': 'truck.n.01',
    'synonyms': ['truck'],
    'def': 'an automotive vehicle suitable for hauling',
    'name': 'truck'
}, {
    'frequency': 'r',
    'id': 1145,
    'synset': 'truffle.n.03',
    'synonyms': ['truffle_(chocolate)', 'chocolate_truffle'],
    'def': 'creamy chocolate candy',
    'name': 'truffle_(chocolate)'
}, {
    'frequency':
        'c',
    'id':
        1146,
    'synset':
        'trunk.n.02',
    'synonyms': ['trunk'],
    'def':
        'luggage consisting of a large strong case used when traveling or for storage',
    'name':
        'trunk'
}, {
    'frequency': 'r',
    'id': 1147,
    'synset': 'tub.n.02',
    'synonyms': ['vat'],
    'def': 'a large open vessel for holding or storing liquids',
    'name': 'vat'
}, {
    'frequency':
        'c',
    'id':
        1148,
    'synset':
        'turban.n.01',
    'synonyms': ['turban'],
    'def':
        'a traditional headdress consisting of a long scarf wrapped around the head',
    'name':
        'turban'
}, {
    'frequency':
        'r',
    'id':
        1149,
    'synset':
        'turkey.n.01',
    'synonyms': ['turkey_(bird)'],
    'def':
        'large gallinaceous bird with fan-shaped tail; widely domesticated for food',
    'name':
        'turkey_(bird)'
}, {
    'frequency': 'c',
    'id': 1150,
    'synset': 'turkey.n.04',
    'synonyms': ['turkey_(food)'],
    'def': 'flesh of large domesticated fowl usually roasted',
    'name': 'turkey_(food)'
}, {
    'frequency':
        'r',
    'id':
        1151,
    'synset':
        'turnip.n.01',
    'synonyms': ['turnip'],
    'def':
        'widely cultivated plant having a large fleshy edible white or yellow root',
    'name':
        'turnip'
}, {
    'frequency':
        'c',
    'id':
        1152,
    'synset':
        'turtle.n.02',
    'synonyms': ['turtle'],
    'def':
        'any of various aquatic and land reptiles having a bony shell and flipper-like limbs for swimming',
    'name':
        'turtle'
}, {
    'frequency': 'r',
    'id': 1153,
    'synset': 'turtleneck.n.01',
    'synonyms': ['turtleneck_(clothing)', 'polo-neck'],
    'def': 'a sweater or jersey with a high close-fitting collar',
    'name': 'turtleneck_(clothing)'
}, {
    'frequency':
        'r',
    'id':
        1154,
    'synset':
        'typewriter.n.01',
    'synonyms': ['typewriter'],
    'def':
        'hand-operated character printer for printing written messages one character at a time',
    'name':
        'typewriter'
}, {
    'frequency': 'f',
    'id': 1155,
    'synset': 'umbrella.n.01',
    'synonyms': ['umbrella'],
    'def': 'a lightweight handheld collapsible canopy',
    'name': 'umbrella'
}, {
    'frequency': 'c',
    'id': 1156,
    'synset': 'underwear.n.01',
    'synonyms': ['underwear', 'underclothes', 'underclothing', 'underpants'],
    'def': 'undergarment worn next to the skin and under the outer garments',
    'name': 'underwear'
}, {
    'frequency': 'r',
    'id': 1157,
    'synset': 'unicycle.n.01',
    'synonyms': ['unicycle'],
    'def': 'a vehicle with a single wheel that is driven by pedals',
    'name': 'unicycle'
}, {
    'frequency':
        'c',
    'id':
        1158,
    'synset':
        'urinal.n.01',
    'synonyms': ['urinal'],
    'def':
        'a plumbing fixture (usually attached to the wall) used by men to urinate',
    'name':
        'urinal'
}, {
    'frequency': 'r',
    'id': 1159,
    'synset': 'urn.n.01',
    'synonyms': ['urn'],
    'def': 'a large vase that usually has a pedestal or feet',
    'name': 'urn'
}, {
    'frequency': 'c',
    'id': 1160,
    'synset': 'vacuum.n.04',
    'synonyms': ['vacuum_cleaner'],
    'def': 'an electrical home appliance that cleans by suction',
    'name': 'vacuum_cleaner'
}, {
    'frequency':
        'c',
    'id':
        1161,
    'synset':
        'valve.n.03',
    'synonyms': ['valve'],
    'def':
        'control consisting of a mechanical device for controlling the flow of a fluid',
    'name':
        'valve'
}, {
    'frequency':
        'f',
    'id':
        1162,
    'synset':
        'vase.n.01',
    'synonyms': ['vase'],
    'def':
        'an open jar of glass or porcelain used as an ornament or to hold flowers',
    'name':
        'vase'
}, {
    'frequency': 'c',
    'id': 1163,
    'synset': 'vending_machine.n.01',
    'synonyms': ['vending_machine'],
    'def': 'a slot machine for selling goods',
    'name': 'vending_machine'
}, {
    'frequency': 'f',
    'id': 1164,
    'synset': 'vent.n.01',
    'synonyms': ['vent', 'blowhole', 'air_vent'],
    'def': 'a hole for the escape of gas or air',
    'name': 'vent'
}, {
    'frequency': 'c',
    'id': 1165,
    'synset': 'videotape.n.01',
    'synonyms': ['videotape'],
    'def': 'a video recording made on magnetic tape',
    'name': 'videotape'
}, {
    'frequency':
        'r',
    'id':
        1166,
    'synset':
        'vinegar.n.01',
    'synonyms': ['vinegar'],
    'def':
        'sour-tasting liquid produced usually by oxidation of the alcohol in wine or cider and used as a condiment or food preservative',
    'name':
        'vinegar'
}, {
    'frequency':
        'r',
    'id':
        1167,
    'synset':
        'violin.n.01',
    'synonyms': ['violin', 'fiddle'],
    'def':
        'bowed stringed instrument that is the highest member of the violin family',
    'name':
        'violin'
}, {
    'frequency': 'r',
    'id': 1168,
    'synset': 'vodka.n.01',
    'synonyms': ['vodka'],
    'def': 'unaged colorless liquor originating in Russia',
    'name': 'vodka'
}, {
    'frequency': 'r',
    'id': 1169,
    'synset': 'volleyball.n.02',
    'synonyms': ['volleyball'],
    'def': 'an inflated ball used in playing volleyball',
    'name': 'volleyball'
}, {
    'frequency':
        'r',
    'id':
        1170,
    'synset':
        'vulture.n.01',
    'synonyms': ['vulture'],
    'def':
        'any of various large birds of prey having naked heads and weak claws and feeding chiefly on carrion',
    'name':
        'vulture'
}, {
    'frequency': 'c',
    'id': 1171,
    'synset': 'waffle.n.01',
    'synonyms': ['waffle'],
    'def': 'pancake batter baked in a waffle iron',
    'name': 'waffle'
}, {
    'frequency': 'r',
    'id': 1172,
    'synset': 'waffle_iron.n.01',
    'synonyms': ['waffle_iron'],
    'def': 'a kitchen appliance for baking waffles',
    'name': 'waffle_iron'
}, {
    'frequency':
        'c',
    'id':
        1173,
    'synset':
        'wagon.n.01',
    'synonyms': ['wagon'],
    'def':
        'any of various kinds of wheeled vehicles drawn by an animal or a tractor',
    'name':
        'wagon'
}, {
    'frequency': 'c',
    'id': 1174,
    'synset': 'wagon_wheel.n.01',
    'synonyms': ['wagon_wheel'],
    'def': 'a wheel of a wagon',
    'name': 'wagon_wheel'
}, {
    'frequency': 'c',
    'id': 1175,
    'synset': 'walking_stick.n.01',
    'synonyms': ['walking_stick'],
    'def': 'a stick carried in the hand for support in walking',
    'name': 'walking_stick'
}, {
    'frequency': 'c',
    'id': 1176,
    'synset': 'wall_clock.n.01',
    'synonyms': ['wall_clock'],
    'def': 'a clock mounted on a wall',
    'name': 'wall_clock'
}, {
    'frequency':
        'f',
    'id':
        1177,
    'synset':
        'wall_socket.n.01',
    'synonyms': [
        'wall_socket', 'wall_plug', 'electric_outlet', 'electrical_outlet',
        'outlet', 'electric_receptacle'
    ],
    'def':
        'receptacle providing a place in a wiring system where current can be taken to run electrical devices',
    'name':
        'wall_socket'
}, {
    'frequency': 'c',
    'id': 1178,
    'synset': 'wallet.n.01',
    'synonyms': ['wallet', 'billfold'],
    'def': 'a pocket-size case for holding papers and paper money',
    'name': 'wallet'
}, {
    'frequency':
        'r',
    'id':
        1179,
    'synset':
        'walrus.n.01',
    'synonyms': ['walrus'],
    'def':
        'either of two large northern marine mammals having ivory tusks and tough hide over thick blubber',
    'name':
        'walrus'
}, {
    'frequency':
        'r',
    'id':
        1180,
    'synset':
        'wardrobe.n.01',
    'synonyms': ['wardrobe'],
    'def':
        'a tall piece of furniture that provides storage space for clothes; has a door and rails or hooks for hanging clothes',
    'name':
        'wardrobe'
}, {
    'frequency':
        'r',
    'id':
        1181,
    'synset':
        'wasabi.n.02',
    'synonyms': ['wasabi'],
    'def':
        'the thick green root of the wasabi plant that the Japanese use in cooking and that tastes like strong horseradish',
    'name':
        'wasabi'
}, {
    'frequency': 'c',
    'id': 1182,
    'synset': 'washer.n.03',
    'synonyms': ['automatic_washer', 'washing_machine'],
    'def': 'a home appliance for washing clothes and linens automatically',
    'name': 'automatic_washer'
}, {
    'frequency': 'f',
    'id': 1183,
    'synset': 'watch.n.01',
    'synonyms': ['watch', 'wristwatch'],
    'def': 'a small, portable timepiece',
    'name': 'watch'
}, {
    'frequency': 'f',
    'id': 1184,
    'synset': 'water_bottle.n.01',
    'synonyms': ['water_bottle'],
    'def': 'a bottle for holding water',
    'name': 'water_bottle'
}, {
    'frequency': 'c',
    'id': 1185,
    'synset': 'water_cooler.n.01',
    'synonyms': ['water_cooler'],
    'def': 'a device for cooling and dispensing drinking water',
    'name': 'water_cooler'
}, {
    'frequency': 'c',
    'id': 1186,
    'synset': 'water_faucet.n.01',
    'synonyms': ['water_faucet', 'water_tap', 'tap_(water_faucet)'],
    'def': 'a faucet for drawing water from a pipe or cask',
    'name': 'water_faucet'
}, {
    'frequency': 'r',
    'id': 1187,
    'synset': 'water_filter.n.01',
    'synonyms': ['water_filter'],
    'def': 'a filter to remove impurities from the water supply',
    'name': 'water_filter'
}, {
    'frequency': 'r',
    'id': 1188,
    'synset': 'water_heater.n.01',
    'synonyms': ['water_heater', 'hot-water_heater'],
    'def': 'a heater and storage tank to supply heated water',
    'name': 'water_heater'
}, {
    'frequency': 'r',
    'id': 1189,
    'synset': 'water_jug.n.01',
    'synonyms': ['water_jug'],
    'def': 'a jug that holds water',
    'name': 'water_jug'
}, {
    'frequency': 'r',
    'id': 1190,
    'synset': 'water_pistol.n.01',
    'synonyms': ['water_gun', 'squirt_gun'],
    'def': 'plaything consisting of a toy pistol that squirts water',
    'name': 'water_gun'
}, {
    'frequency':
        'c',
    'id':
        1191,
    'synset':
        'water_scooter.n.01',
    'synonyms': ['water_scooter', 'sea_scooter', 'jet_ski'],
    'def':
        'a motorboat resembling a motor scooter (NOT A SURFBOARD OR WATER SKI)',
    'name':
        'water_scooter'
}, {
    'frequency':
        'c',
    'id':
        1192,
    'synset':
        'water_ski.n.01',
    'synonyms': ['water_ski'],
    'def':
        'broad ski for skimming over water towed by a speedboat (DO NOT MARK WATER)',
    'name':
        'water_ski'
}, {
    'frequency': 'c',
    'id': 1193,
    'synset': 'water_tower.n.01',
    'synonyms': ['water_tower'],
    'def': 'a large reservoir for water',
    'name': 'water_tower'
}, {
    'frequency':
        'c',
    'id':
        1194,
    'synset':
        'watering_can.n.01',
    'synonyms': ['watering_can'],
    'def':
        'a container with a handle and a spout with a perforated nozzle; used to sprinkle water over plants',
    'name':
        'watering_can'
}, {
    'frequency':
        'c',
    'id':
        1195,
    'synset':
        'watermelon.n.02',
    'synonyms': ['watermelon'],
    'def':
        'large oblong or roundish melon with a hard green rind and sweet watery red or occasionally yellowish pulp',
    'name':
        'watermelon'
}, {
    'frequency':
        'f',
    'id':
        1196,
    'synset':
        'weathervane.n.01',
    'synonyms': ['weathervane', 'vane_(weathervane)', 'wind_vane'],
    'def':
        'mechanical device attached to an elevated structure; rotates freely to show the direction of the wind',
    'name':
        'weathervane'
}, {
    'frequency':
        'c',
    'id':
        1197,
    'synset':
        'webcam.n.01',
    'synonyms': ['webcam'],
    'def':
        'a digital camera designed to take digital photographs and transmit them over the internet',
    'name':
        'webcam'
}, {
    'frequency':
        'c',
    'id':
        1198,
    'synset':
        'wedding_cake.n.01',
    'synonyms': ['wedding_cake', 'bridecake'],
    'def':
        'a rich cake with two or more tiers and covered with frosting and decorations; served at a wedding reception',
    'name':
        'wedding_cake'
}, {
    'frequency': 'c',
    'id': 1199,
    'synset': 'wedding_ring.n.01',
    'synonyms': ['wedding_ring', 'wedding_band'],
    'def': 'a ring given to the bride and/or groom at the wedding',
    'name': 'wedding_ring'
}, {
    'frequency':
        'f',
    'id':
        1200,
    'synset':
        'wet_suit.n.01',
    'synonyms': ['wet_suit'],
    'def':
        'a close-fitting garment made of a permeable material; worn in cold water to retain body heat',
    'name':
        'wet_suit'
}, {
    'frequency':
        'f',
    'id':
        1201,
    'synset':
        'wheel.n.01',
    'synonyms': ['wheel'],
    'def':
        'a circular frame with spokes (or a solid disc) that can rotate on a shaft or axle',
    'name':
        'wheel'
}, {
    'frequency': 'c',
    'id': 1202,
    'synset': 'wheelchair.n.01',
    'synonyms': ['wheelchair'],
    'def': 'a movable chair mounted on large wheels',
    'name': 'wheelchair'
}, {
    'frequency': 'c',
    'id': 1203,
    'synset': 'whipped_cream.n.01',
    'synonyms': ['whipped_cream'],
    'def': 'cream that has been beaten until light and fluffy',
    'name': 'whipped_cream'
}, {
    'frequency': 'r',
    'id': 1204,
    'synset': 'whiskey.n.01',
    'synonyms': ['whiskey'],
    'def': 'a liquor made from fermented mash of grain',
    'name': 'whiskey'
}, {
    'frequency':
        'r',
    'id':
        1205,
    'synset':
        'whistle.n.03',
    'synonyms': ['whistle'],
    'def':
        'a small wind instrument that produces a whistling sound by blowing into it',
    'name':
        'whistle'
}, {
    'frequency': 'r',
    'id': 1206,
    'synset': 'wick.n.02',
    'synonyms': ['wick'],
    'def': 'a loosely woven cord in a candle or oil lamp that is lit on fire',
    'name': 'wick'
}, {
    'frequency': 'c',
    'id': 1207,
    'synset': 'wig.n.01',
    'synonyms': ['wig'],
    'def': 'hairpiece covering the head and made of real or synthetic hair',
    'name': 'wig'
}, {
    'frequency':
        'c',
    'id':
        1208,
    'synset':
        'wind_chime.n.01',
    'synonyms': ['wind_chime'],
    'def':
        'a decorative arrangement of pieces of metal or glass or pottery that hang together loosely so the wind can cause them to tinkle',
    'name':
        'wind_chime'
}, {
    'frequency': 'c',
    'id': 1209,
    'synset': 'windmill.n.01',
    'synonyms': ['windmill'],
    'def': 'a mill that is powered by the wind',
    'name': 'windmill'
}, {
    'frequency': 'c',
    'id': 1210,
    'synset': 'window_box.n.01',
    'synonyms': ['window_box_(for_plants)'],
    'def': 'a container for growing plants on a windowsill',
    'name': 'window_box_(for_plants)'
}, {
    'frequency': 'f',
    'id': 1211,
    'synset': 'windshield_wiper.n.01',
    'synonyms': [
        'windshield_wiper', 'windscreen_wiper', 'wiper_(for_windshield/screen)'
    ],
    'def': 'a mechanical device that cleans the windshield',
    'name': 'windshield_wiper'
}, {
    'frequency':
        'c',
    'id':
        1212,
    'synset':
        'windsock.n.01',
    'synonyms': [
        'windsock', 'air_sock', 'air-sleeve', 'wind_sleeve', 'wind_cone'
    ],
    'def':
        'a truncated cloth cone mounted on a mast/pole; shows wind direction',
    'name':
        'windsock'
}, {
    'frequency': 'f',
    'id': 1213,
    'synset': 'wine_bottle.n.01',
    'synonyms': ['wine_bottle'],
    'def': 'a bottle for holding wine',
    'name': 'wine_bottle'
}, {
    'frequency': 'r',
    'id': 1214,
    'synset': 'wine_bucket.n.01',
    'synonyms': ['wine_bucket', 'wine_cooler'],
    'def': 'a bucket of ice used to chill a bottle of wine',
    'name': 'wine_bucket'
}, {
    'frequency': 'f',
    'id': 1215,
    'synset': 'wineglass.n.01',
    'synonyms': ['wineglass'],
    'def': 'a glass that has a stem and in which wine is served',
    'name': 'wineglass'
}, {
    'frequency': 'r',
    'id': 1216,
    'synset': 'wing_chair.n.01',
    'synonyms': ['wing_chair'],
    'def': 'easy chair having wings on each side of a high back',
    'name': 'wing_chair'
}, {
    'frequency': 'c',
    'id': 1217,
    'synset': 'winker.n.02',
    'synonyms': ['blinder_(for_horses)'],
    'def': 'blinds that prevent a horse from seeing something on either side',
    'name': 'blinder_(for_horses)'
}, {
    'frequency': 'c',
    'id': 1218,
    'synset': 'wok.n.01',
    'synonyms': ['wok'],
    'def': 'pan with a convex bottom; used for frying in Chinese cooking',
    'name': 'wok'
}, {
    'frequency':
        'r',
    'id':
        1219,
    'synset':
        'wolf.n.01',
    'synonyms': ['wolf'],
    'def':
        'a wild carnivorous mammal of the dog family, living and hunting in packs',
    'name':
        'wolf'
}, {
    'frequency': 'c',
    'id': 1220,
    'synset': 'wooden_spoon.n.02',
    'synonyms': ['wooden_spoon'],
    'def': 'a spoon made of wood',
    'name': 'wooden_spoon'
}, {
    'frequency': 'c',
    'id': 1221,
    'synset': 'wreath.n.01',
    'synonyms': ['wreath'],
    'def': 'an arrangement of flowers, leaves, or stems fastened in a ring',
    'name': 'wreath'
}, {
    'frequency': 'c',
    'id': 1222,
    'synset': 'wrench.n.03',
    'synonyms': ['wrench', 'spanner'],
    'def': 'a hand tool that is used to hold or twist a nut or bolt',
    'name': 'wrench'
}, {
    'frequency': 'c',
    'id': 1223,
    'synset': 'wristband.n.01',
    'synonyms': ['wristband'],
    'def': 'band consisting of a part of a sleeve that covers the wrist',
    'name': 'wristband'
}, {
    'frequency': 'f',
    'id': 1224,
    'synset': 'wristlet.n.01',
    'synonyms': ['wristlet', 'wrist_band'],
    'def': 'a band or bracelet worn around the wrist',
    'name': 'wristlet'
}, {
    'frequency':
        'r',
    'id':
        1225,
    'synset':
        'yacht.n.01',
    'synonyms': ['yacht'],
    'def':
        'an expensive vessel propelled by sail or power and used for cruising or racing',
    'name':
        'yacht'
}, {
    'frequency': 'r',
    'id': 1226,
    'synset': 'yak.n.02',
    'synonyms': ['yak'],
    'def': 'large long-haired wild ox of Tibet often domesticated',
    'name': 'yak'
}, {
    'frequency': 'c',
    'id': 1227,
    'synset': 'yogurt.n.01',
    'synonyms': ['yogurt', 'yoghurt', 'yoghourt'],
    'def': 'a custard-like food made from curdled milk',
    'name': 'yogurt'
}, {
    'frequency': 'r',
    'id': 1228,
    'synset': 'yoke.n.07',
    'synonyms': ['yoke_(animal_equipment)'],
    'def': 'gear joining two animals at the neck; NOT egg yolk',
    'name': 'yoke_(animal_equipment)'
}, {
    'frequency': 'f',
    'id': 1229,
    'synset': 'zebra.n.01',
    'synonyms': ['zebra'],
    'def': 'any of several fleet black-and-white striped African equines',
    'name': 'zebra'
}, {
    'frequency': 'c',
    'id': 1230,
    'synset': 'zucchini.n.02',
    'synonyms': ['zucchini', 'courgette'],
    'def': 'small cucumber-shaped vegetable marrow; typically dark green',
    'name': 'zucchini'
}]  # noqa
# fmt: on
