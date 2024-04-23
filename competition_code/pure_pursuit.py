import numpy as np
import math


class PurePursuit():
    def __init__(self, path):
        self.path = path

    def sgn (self, x):
        if x < 0:
            return -1
        else:
            return 1
    
    def pure_pursuit_step(self, currentPos, currentHeading, lookAheadDis, LFindex):
        # extract currentX and currentY
        currentX = currentPos[0]
        currentY = currentPos[1]

        # use for loop to search intersections
        lastFoundIndex = LFindex
        intersectFound = False
        startingIndex = lastFoundIndex

        for i in range(startingIndex, len(self.path) - 1):
            # beginning of line-circle intersection code
            x1 = self.path[i][0] - currentX
            y1 = self.path[i][1] - currentY
            x2 = self.path[i + 1][0] - currentX
            y2 = self.path[i + 1][1] - currentY
            dx = x2 - x1
            dy = y2 - y1
            dr = math.sqrt(dx**2 + dy**2)
            D = x1 * y2 - x2 * y1
            discriminant = (lookAheadDis**2) * (dr**2) - D**2

            if discriminant >= 0:
                sol_x1 = (D * dy + self.sgn(dy) * dx * np.sqrt(discriminant)) / dr**2
                sol_x2 = (D * dy - self.sgn(dy) * dx * np.sqrt(discriminant)) / dr**2
                sol_y1 = (-D * dx + abs(dy) * np.sqrt(discriminant)) / dr**2
                sol_y2 = (-D * dx - abs(dy) * np.sqrt(discriminant)) / dr**2

                sol_pt1 = [sol_x1 + currentX, sol_y1 + currentY]
                sol_pt2 = [sol_x2 + currentX, sol_y2 + currentY]
                # end of line-circle intersection code

                minX = min(self.path[i][0], self.path[i + 1][0])
                minY = min(self.path[i][1], self.path[i + 1][1])
                maxX = max(self.path[i][0], self.path[i + 1][0])
                maxY = max(self.path[i][1], self.path[i + 1][1])

                # if one or both of the solutions are in range
                if ((minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY)) or (
                    (minX <= sol_pt2[0] <= maxX) and (minY <= sol_pt2[1] <= maxY)
                ):
                    foundIntersection = True

                    # if both solutions are in range, check which one is better
                    if ((minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY)) and (
                        (minX <= sol_pt2[0] <= maxX) and (minY <= sol_pt2[1] <= maxY)
                    ):
                        # make the decision by compare the distance between the intersections and the next point in path
                        if np.linalg.norm(sol_pt1, self.path[i + 1]) < np.linalg.norm(sol_pt2, self.path[i + 1]):
                            goalPt = sol_pt1
                        else:
                            goalPt = sol_pt2

                    # if not both solutions are in range, take the one that's in range
                    else:
                        # if solution pt1 is in range, set that as goal point
                        if (minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY):
                            goalPt = sol_pt1
                        else:
                            goalPt = sol_pt2

                    # only exit loop if the solution pt found is closer to the next pt in path than the current pos
                    if np.linalg.norm(goalPt, self.path[i + 1]) < np.linalg.norm([currentX, currentY], self.path[i + 1]):
                        # update lastFoundIndex and exit
                        lastFoundIndex = i
                        break
                    else:
                        # in case for some reason the robot cannot find intersection in the next path segment, but we also don't want it to go backward
                        lastFoundIndex = i + 1

                # if no solutions are in range
                else:
                    foundIntersection = False
                    # no new intersection found, potentially deviated from the path
                    # follow path[lastFoundIndex]
                    goalPt = [self.path[lastFoundIndex][0], self.path[lastFoundIndex][1]]

            # if determinant < 0
            else:
                foundIntersection = False
                # no new intersection found, potentially deviated from the path
                # follow path[lastFoundIndex]
                goalPt = [self.path[lastFoundIndex][0], self.path[lastFoundIndex][1]]

        # obtained goal point, now compute turn vel
        # initialize proportional controller constant
        Kp = 5

        # calculate absTargetAngle with the atan2 function
        absTargetAngle = (
            math.atan2(goalPt[1] - currentPos[1], goalPt[0] - currentPos[0]) * 180 / pi
        )
        if absTargetAngle < 0:
            absTargetAngle += 360

        # compute turn error by finding the minimum angle
        turnError = absTargetAngle - currentHeading
        if turnError > 180 or turnError < -180:
            turnError = -1 * self.sgn(turnError) * (360 - abs(turnError))

        # apply proportional controller
        turnVel = Kp * turnError

        return goalPt, lastFoundIndex, turnVel
