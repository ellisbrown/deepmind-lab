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

local make_map = require 'common.make_map'
local vlr_objects = require 'common.vlr_objects'
local custom_observations = require 'decorators.custom_observations'
local game = require 'dmlab.system.game'
local timeout = require 'decorators.timeout'
local api = {}

local MAP_ENTITIES = [[
************
*P*       e*
* *        *
* *   a    *
*          *
*   d     f*
************
]]


-- **********
-- *P*d    e*
-- * *      *
-- *        *
-- *     f  *
-- **********

-- More complex map, not sure if agent handles this well though...
-- ************
-- *P*a*b*c  i*
-- * * * *    *
-- *          *
-- * * * *    *
-- * *d*e*f  h*
-- ************



function api:init(params)
  make_map.seedRng(1)
  api._map = make_map.makeMap{
      mapName = "custom env1",
      mapEntityLayer = MAP_ENTITIES,
      useSkybox = false,
        pickups = {
--             A = 'apple2',
--             B = 'ball',
--             C = 'balloon',
--             D = 'banana',
--             E = 'bottle',
--             F = 'cake',
--             G = 'hr_knife',
--             H = 'hr_ladder',
--             I = 'hr_pencil',
--             J = 'hr_hair_brush',
--             K = 'hr_cow',
                a = 'apple2',
                b = 'balloon',
                c = 'banana',
                d = 'hr_cow',
                e = 'hr_tv',
                f = 'hr_pencil',
                h = 'hr_hair_brush',
                i = 'hr_spoon',


                C = 'hr_apple2',
                D = 'hr_key_lrg',
                E = 'hr_cassette',
                G = 'hr_spoon',
                H = 'hr_hammer',
                I = 'hr_tree',
                J = 'hr_car',
                L = 'hr_pincer',
                M = 'hr_saxophone',
                N = 'hr_hat',
                O = 'hr_cake',
--                 P = 'hr_tennis_racket',  P represents the player, cannot overwrite
                Q = 'hr_fork',
                R = 'hr_shoe',
                S = 'hr_can',
                T = 'hr_jug',
                U = 'hr_ice_lolly',
                V = 'hr_ice_lolly_lrg',
                W = 'hr_mug',
                X = 'hr_ball',
                Y = 'hr_balloon',
                Z = 'hr_suitcase',
        }
  }
end

function api:createPickup(className)
return vlr_objects.defaults[className]
end

function api:canPickup(id)
  -- Turn off pickups so you can get up close and personal.
  return false
end

function api:nextMap()
  return self._map
end

function api:updateSpawnVars(spawnVars)
  if spawnVars.classname == "info_player_start" then
    -- Spawn facing East.
    spawnVars.angle = "0"
    spawnVars.randomAngleRange = "0"
  end
  return spawnVars
end

timeout.decorate(api, 60 * 60)
custom_observations.decorate(api)

return api
