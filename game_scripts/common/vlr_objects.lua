--[[ Copyright (C) 2018 Google Inc.

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
]]

local pickups = {}

-- Must match itemType_t in engine/code/game/bg_public.h.
pickups.type = {
    INVALID = 0,
    WEAPON = 1,
    AMMO = 2,
    ARMOR = 3,
    HEALTH = 4,
    POWER_UP = 5,
    HOLDABLE = 6,
    PERSISTANT_POWERUP = 7,
    TEAM = 8,
    REWARD = 9,
    GOAL = 10
}

-- Must match reward_mv_t in engine/code/game/bg_public.h.
pickups.moveType = {
    BOB = 0,
    STATIC = 1
}

pickups.defaults = {
    apple = {
        name = 'Apple',
        classname = 'apple',
        model = 'models/apple.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
    lemon = {
        name = 'Lemon',
        classname = 'lemon',
        model = 'models/lemon.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
    strawberry = {
        name = 'Strawberry',
        classname = 'strawberry',
        model = 'models/strawberry.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
    fungi = {
        name = 'Fungi',
        classname = 'fungi',
        model = 'models/toadstool.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
    watermelon = {
        name = 'Watermelon',
        classname = 'watermelon',
        model = 'models/watermelon.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
    mango = {
        name = 'Mango',
        classname = 'mango',
        model = 'models/mango.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
hr_pencil = {
        name = 'hr_pencil',
        classname = 'hr_pencil',
        model = 'models/hr_pencil.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
hr_hair_brush = {
        name = 'hr_hair_brush',
        classname = 'hr_hair_brush',
        model = 'models/hr_hair_brush.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
hr_apple2 = {
        name = 'hr_apple2',
        classname = 'hr_apple2',
        model = 'models/hr_apple2.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
hr_key_lrg = {
        name = 'hr_key_lrg',
        classname = 'hr_key_lrg',
        model = 'models/hr_key_lrg.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
hr_cassette = {
        name = 'hr_cassette',
        classname = 'hr_cassette',
        model = 'models/hr_cassette.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
hr_cow = {
        name = 'hr_cow',
        classname = 'hr_cow',
        model = 'models/hr_cow.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
hr_spoon = {
        name = 'hr_spoon',
        classname = 'hr_spoon',
        model = 'models/hr_spoon.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
hr_hammer = {
        name = 'hr_hammer',
        classname = 'hr_hammer',
        model = 'models/hr_hammer.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
hr_tree = {
        name = 'hr_tree',
        classname = 'hr_tree',
        model = 'models/hr_tree.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
hr_car = {
        name = 'hr_car',
        classname = 'hr_car',
        model = 'models/hr_car.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
hr_tv = {
        name = 'hr_tv',
        classname = 'hr_tv',
        model = 'models/hr_tv.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
hr_pincer = {
        name = 'hr_pincer',
        classname = 'hr_pincer',
        model = 'models/hr_pincer.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
hr_saxophone = {
        name = 'hr_saxophone',
        classname = 'hr_saxophone',
        model = 'models/hr_saxophone.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
hr_hat = {
        name = 'hr_hat',
        classname = 'hr_hat',
        model = 'models/hr_hat.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
hr_cake = {
        name = 'hr_cake',
        classname = 'hr_cake',
        model = 'models/hr_cake.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
hr_tennis_racket = {
        name = 'hr_tennis_racket',
        classname = 'hr_tennis_racket',
        model = 'models/hr_tennis_racket.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
hr_fork = {
        name = 'hr_fork',
        classname = 'hr_fork',
        model = 'models/hr_fork.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
hr_shoe = {
        name = 'hr_shoe',
        classname = 'hr_shoe',
        model = 'models/hr_shoe.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
hr_can = {
        name = 'hr_can',
        classname = 'hr_can',
        model = 'models/hr_can.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
hr_jug = {
        name = 'hr_jug',
        classname = 'hr_jug',
        model = 'models/hr_jug.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
hr_ice_lolly = {
        name = 'hr_ice_lolly',
        classname = 'hr_ice_lolly',
        model = 'models/hr_ice_lolly.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
hr_ice_lolly_lrg = {
        name = 'hr_ice_lolly_lrg',
        classname = 'hr_ice_lolly_lrg',
        model = 'models/hr_ice_lolly_lrg.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
hr_mug = {
        name = 'hr_mug',
        classname = 'hr_mug',
        model = 'models/hr_mug.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
hr_ball = {
        name = 'hr_ball',
        classname = 'hr_ball',
        model = 'models/hr_ball.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
hr_balloon = {
        name = 'hr_balloon',
        classname = 'hr_balloon',
        model = 'models/hr_balloon.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
hr_suitcase = {
        name = 'hr_suitcase',
        classname = 'hr_suitcase',
        model = 'models/hr_suitcase.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
hr_tomato = {
        name = 'hr_tomato',
        classname = 'hr_tomato',
        model = 'models/hr_tomato.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
hr_pig = {
        name = 'hr_pig',
        classname = 'hr_pig',
        model = 'models/hr_pig.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
hr_guitar = {
        name = 'hr_guitar',
        classname = 'hr_guitar',
        model = 'models/hr_guitar.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
hr_cherries = {
        name = 'hr_cherries',
        classname = 'hr_cherries',
        model = 'models/hr_cherries.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
hr_bee = {
        name = 'hr_bee',
        classname = 'hr_bee',
        model = 'models/hr_bee.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
hr_chair = {
        name = 'hr_chair',
        classname = 'hr_chair',
        model = 'models/hr_chair.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
hr_zebra = {
        name = 'hr_zebra',
        classname = 'hr_zebra',
        model = 'models/hr_zebra.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
hr_banana = {
        name = 'hr_banana',
        classname = 'hr_banana',
        model = 'models/hr_banana.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
hr_wine_glass = {
        name = 'hr_wine_glass',
        classname = 'hr_wine_glass',
        model = 'models/hr_wine_glass.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
hr_flower = {
        name = 'hr_flower',
        classname = 'hr_flower',
        model = 'models/hr_flower.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
hr_key = {
        name = 'hr_key',
        classname = 'hr_key',
        model = 'models/hr_key.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
hr_toothbrush = {
        name = 'hr_toothbrush',
        classname = 'hr_toothbrush',
        model = 'models/hr_toothbrush.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
hr_fridge = {
        name = 'hr_fridge',
        classname = 'hr_fridge',
        model = 'models/hr_fridge.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
hr_plant = {
        name = 'hr_plant',
        classname = 'hr_plant',
        model = 'models/hr_plant.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
hr_bottle = {
        name = 'hr_bottle',
        classname = 'hr_bottle',
        model = 'models/hr_bottle.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
hr_ladder = {
        name = 'hr_ladder',
        classname = 'hr_ladder',
        model = 'models/hr_ladder.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },
hr_knife = {
        name = 'hr_knife',
        classname = 'hr_knife',
        model = 'models/hr_knife.md3',
        quantity = 1,
        type = pickups.type.PERSISTANT_POWERUP
    },



}

return pickups
