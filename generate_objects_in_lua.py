import os
all_objects = os.listdir("assets/models")
all_objects = [fname[:-4] for fname in all_objects
               if fname.endswith(".md3") and fname.startswith("hr_")]

format = """%s = {
        name = '%s',
        classname = '%s',
        model = 'models/%s.md3',
        quantity = 1,
        type = pickups.type.REWARD
    },"""

# Print out letter to object string mapping, used directly in each env
# for example: deepmind-lab/game_scripts/levels/vlr_env1.lua
letters = list(range(ord('A'), ord('Z') + 1))
for i in range(len(letters)):
    print("%s = '%s'," % (chr(letters[i]), all_objects[i]))


# Populate new objects in "pickups.defaults" of deepmind-lab/game_scripts/common/vlr_objects.lua
# for name in all_objects:
#     new_obj_text = format % (name, name, name, name)
#     print(new_obj_text)
