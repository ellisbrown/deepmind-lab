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
local pickups = require 'common.pickups'
local vlr_objects = require 'common.vlr_objects'
local custom_observations = require 'decorators.custom_observations'
local game = require 'dmlab.system.game'
local timeout = require 'decorators.timeout'
local api = {}

local open = io.open

local function read_file(path)
    local file = open(path, "r") -- r read mode 
    if not file then return nil end
    local content = file:read "*a" -- *a or *all reads the whole file
    file:close()
    return content
end


local object_map = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O"}
local angles = {"270", "90", "0", "180", "315", "135", "45", "225", "0"}
-- local angles = {"90", "90", "0", "180", "315", "135", "45", "225"}

function api:init(params)
  make_map.seedRng(1)
  api._count = 0
end

function api:createPickup(className)
  return vlr_objects.defaults[className]
end
  
function api:canPickup(id)
  -- Turn off pickups so you can get up close and personal.
  return false
end


function api:nextMap()
  local obj_id = math.floor(api._count / 32) + 1
  local object = object_map[obj_id]
  local map_id = api._count % 32
  local path = "./game_scripts/levels/vlr_maps/vlr_maps_" .. object .. "_" .. map_id .. ".txt"  
  local mapText = read_file(path)
  api._map = make_map.makeMap{
    mapName = 'vlr_maps_' .. object .. '_' .. map_id,
    mapEntityLayer = mapText,
    useSkybox = true,
    pickups = {
      A = 'hr_pencil',
      B = 'hr_hair_brush',
      C = 'hr_apple2',
      D = 'hr_key_lrg',
      E = 'hr_cassette',
      F = 'hr_cow',
      G = 'hr_spoon',
      H = 'hr_hammer',
      I = 'hr_tree',
      J = 'hr_car',
      K = 'hr_tv',
      L = 'hr_pincer',
      M = 'hr_saxophone',
      N = 'hr_hat',
      O = 'hr_cake',
    },
  }
  api._count = api._count + 1

  return api._map
end


function api:updateSpawnVars(spawnVars)
  spawnVars.angle = angles[(api._count - 1) % 8 + 1]
  spawnVars.randomAngleRange = "0"
  return spawnVars
end

timeout.decorate(api, 60 * 60)
custom_observations.decorate(api)

return api


  
