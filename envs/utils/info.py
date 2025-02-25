class Timeout(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Timeout'


class ReachGoal(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Reaching goal'


class Danger(object):
    def __init__(self):
        # self.min_dist = min_dist
        pass 

    def __str__(self):
        return 'Too close'


class Collision(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Collision'


class Nothing(object):
    def __init__(self):
        pass

    def __str__(self):
        return ''
