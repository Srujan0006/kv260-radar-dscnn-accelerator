#ifndef RADAR_CLASSES_H
#define RADAR_CLASSES_H

typedef enum {
    RADAR_CLASS_DRONE = 0,
    RADAR_CLASS_BIRD = 1,
    RADAR_CLASS_HUMAN = 2,
    RADAR_CLASS_CORNER_REFLECTOR = 3
} radar_class_id_t;

static inline const char *radar_class_name(int class_id)
{
    static const char *const names[4] = {
        "Drone", "Bird", "Human", "CornerReflector"
    };
    return (class_id >= 0 && class_id < 4) ? names[class_id] : "Unknown";
}

#endif
