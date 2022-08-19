import numpy as np
import math
# Constants
LegLength = 15                      
MaxLegForce = 0.05
ForwardAngleLimit = np.pi/6.0
BackwardAngleLimit = -np.pi/6.0
MaxVelocity = 6.0
MaxTorque = 0.5
MaxOmega = 1.0

class LeggedAgent:

    def __init__(self):
        self.cx = 0.0                                               # X-position of the agent
        self.cy = 0.0                                               # Y-position of the agent
        self.vx = 0.0                                               # Velocity of the agent
        self.footstate = 0                                          # State of the foot
        self.angle = ForwardAngleLimit                              # Leg angle
        self.omega = 0.0                                            # Leg angular velocity
        self.forwardForce = 0.0                                     # Forward force applied to leg
        self.backwardForce = 0.0                                    # Backward force applied to leg
        self.jointX = self.cx                                       # X-position of the leg joint
        self.jointY = self.cy + 12.5                                # Y-position of the leg joint
        self.footX = self.jointX + LegLength * np.sin(self.angle)   # X-position of the foot
        self.footY = self.jointY + LegLength * np.cos(self.angle)   # Y-position of the foot

    def state(self):
        return np.array([self.angle, self.omega, self.footstate])

    def anglefeedback(self):
        return self.angle * 5.0/ForwardAngleLimit

    def step3(self, stepsize, u):

        force = 0.0

        # Update the leg effectors
        if (u[0] > 0.5):
            self.footstate = 1
            self.omega = 0
        else:
            self.footstate = 0

        self.forwardForce = u[1] * MaxLegForce
        self.backwardForce = u[2] * MaxLegForce

        # Compute force applied to the body
        f = self.forwardForce - self.backwardForce
        if (self.footstate == 1.0):
            if ((self.angle >= BackwardAngleLimit and self.angle <= ForwardAngleLimit) or
                (self.angle < BackwardAngleLimit and f < 0) or
                (self.angle > ForwardAngleLimit and f > 0)):
                force = f

        # Update the position of the body
        self.vx = self.vx + stepsize * force
        if (self.vx < -MaxVelocity):
            self.vx = -MaxVelocity
        if (self.vx > MaxVelocity):
            self.vx = MaxVelocity
        self.cx = self.cx + stepsize * self.vx

        # Update the leg geometry
        self.jointX = self.jointX + stepsize * self.vx
        if (self.footstate == 1.0):
            angle = np.arctan2(self.footX - self.jointX, self.footY - self.jointY)
            self.omega = (angle - self.angle)/stepsize
            self.angle = angle
        else:
            self.vx = 0.0
            self.omega = self.omega + stepsize * MaxTorque * (self.backwardForce - self.forwardForce)
            if (self.omega < -MaxOmega):
                self.omega = -MaxOmega
            if (self.omega > MaxOmega):
                self.omega = MaxOmega
            self.angle = self.angle + stepsize * self.omega
            if (self.angle < BackwardAngleLimit):
                self.angle = BackwardAngleLimit
                self.omega = 0
            if (self.angle > ForwardAngleLimit):
                self.angle = ForwardAngleLimit
                self.omega = 0
            self.footX = self.jointX + LegLength * np.sin(self.angle)
            self.footY = self.jointY + LegLength * np.cos(self.angle)

        # If the foot is too far back, the body becomes "unstable" and forward motion ceases
        if (self.cx - self.footX > 20):
            self.vx = 0.0

    def step2(self, stepsize, u):
        force = 0.0
        # Update the leg effectors
        if (u[0] > 0.5):
            self.footstate = 1
            self.omega = 0
        else:
            self.footstate = 0
        self.forwardForce = u[0] * MaxLegForce
        self.backwardForce = u[1] * MaxLegForce
        # Compute force applied to the body
        f = self.forwardForce - self.backwardForce
        if (self.footstate == 1.0):
            if ((self.angle >= BackwardAngleLimit and self.angle <= ForwardAngleLimit) or
                (self.angle < BackwardAngleLimit and f < 0) or
                (self.angle > ForwardAngleLimit and f > 0)):
                force = f
        # Update the position of the body
        self.vx = self.vx + stepsize * force
        if (self.vx < -MaxVelocity):
            self.vx = -MaxVelocity
        if (self.vx > MaxVelocity):
            self.vx = MaxVelocity
        self.cx = self.cx + stepsize * self.vx
        # Update the leg geometry
        self.jointX = self.jointX + stepsize * self.vx
        if (self.footstate == 1.0):
            angle = math.atan2(self.footX - self.jointX, self.footY - self.jointY)
            self.omega = (angle - self.angle)/stepsize
            self.angle = angle
        else:
            self.vx = 0.0
            self.omega = self.omega + stepsize * MaxTorque * (self.backwardForce - self.forwardForce)
            if (self.omega < -MaxOmega):
                self.omega = -MaxOmega
            if (self.omega > MaxOmega):
                self.omega = MaxOmega
            self.angle = self.angle + stepsize * self.omega
            if (self.angle < BackwardAngleLimit):
                self.angle = BackwardAngleLimit
                self.omega = 0
            if (self.angle > ForwardAngleLimit):
                self.angle = ForwardAngleLimit
                self.omega = 0
            self.footX = self.jointX + LegLength * math.sin(self.angle)
            self.footY = self.jointY + LegLength * math.cos(self.angle)
        # If the foot is too far back, the body becomes "unstable" and forward motion ceases
        if (self.cx - self.footX > 20):
            self.vx = 0.0

    def step1(self, stepsize, u):
        force = 0.0
        # Update the leg effectors
        if (u[0] > 0.5):
            self.footstate = 1
            self.omega = 0
            self.forwardForce = 2 * (u[0] - 0.5) * MaxLegForce;
            self.backwardForce = 0.0
        else:
            self.footstate = 0
            self.forwardForce = 0.0
            self.backwardForce = 2 * (0.5 - u[0]) * MaxLegForce;

        # Compute force applied to the body
        f = self.forwardForce - self.backwardForce
        if (self.footstate == 1.0):
            if ((self.angle >= BackwardAngleLimit and self.angle <= ForwardAngleLimit) or
                (self.angle < BackwardAngleLimit and f < 0) or
                (self.angle > ForwardAngleLimit and f > 0)):
                force = f

        # Update the position of the body
        self.vx = self.vx + stepsize * force
        if (self.vx < -MaxVelocity):
            self.vx = -MaxVelocity
        if (self.vx > MaxVelocity):
            self.vx = MaxVelocity
        self.cx = self.cx + stepsize * self.vx

        # Update the leg geometry
        self.jointX = self.jointX + stepsize * self.vx
        if (self.footstate == 1.0):
            angle = math.atan2(self.footX - self.jointX, self.footY - self.jointY)
            self.omega = (angle - self.angle)/stepsize
            self.angle = angle
        else:
            self.vx = 0.0
            self.omega = self.omega + stepsize * MaxTorque * (self.backwardForce - self.forwardForce)
            if (self.omega < -MaxOmega):
                self.omega = -MaxOmega
            if (self.omega > MaxOmega):
                self.omega = MaxOmega
            self.angle = self.angle + stepsize * self.omega
            if (self.angle < BackwardAngleLimit):
                self.angle = BackwardAngleLimit
                self.omega = 0
            if (self.angle > ForwardAngleLimit):
                self.angle = ForwardAngleLimit
                self.omega = 0
            self.footX = self.jointX + LegLength * math.sin(self.angle)
            self.footY = self.jointY + LegLength * math.cos(self.angle)

        # If the foot is too far back, the body becomes "unstable" and forward motion ceases
        if (self.cx - self.footX > 20):
            self.vx = 0.0
