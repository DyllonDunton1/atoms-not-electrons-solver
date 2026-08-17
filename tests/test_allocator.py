"""Tests for the baseline FIFO task queue and allocator."""

import unittest

from src.allocator import TaskAllocator, TaskQueue
from src.models import Robot
from src.tasks import Task


class TestTaskQueue(unittest.TestCase):
    def test_tasks_zero_through_nine_pop_in_exact_fifo_order(self):
        queue = TaskQueue()
        for task_id in range(10):
            queue.push(Task(task_id))

        popped_ids = [queue.pop().task_id for _ in range(10)]

        self.assertEqual(popped_ids, list(range(10)))
        self.assertEqual(len(queue), 0)
        self.assertIsNone(queue.pop())

    def test_queue_length_decreases_once_per_pop(self):
        queue = TaskQueue()
        for task_id in range(4):
            queue.push(Task(task_id))

        self.assertEqual(len(queue), 4)
        for expected_length in [3, 2, 1, 0]:
            self.assertIsNotNone(queue.pop())
            self.assertEqual(len(queue), expected_length)

        # Repeated reads of an empty queue stay empty and do not produce tasks.
        self.assertIsNone(queue.pop())
        self.assertIsNone(queue.pop())
        self.assertEqual(len(queue), 0)


class TestTaskAllocator(unittest.TestCase):
    def setUp(self):
        self.allocator = TaskAllocator()
        self.robots = {
            robot_id: Robot(robot_id, (robot_id, 0))
            for robot_id in range(5)
        }

    def test_assignment_order_follows_fifo_not_robot_id(self):
        queue = TaskQueue()
        for task_id in range(10):
            queue.push(Task(task_id))

        # This represents robots becoming idle at different times. Robot 4 is
        # first to request work, Robot 0 is not first, and robots may become
        # available more than once over the lifetime of the queue.
        idle_order = [4, 1, 4, 3, 0, 2, 1, 3, 4, 0]
        assignments = []

        for robot_id in idle_order:
            task = self.allocator.assign_next(self.robots[robot_id], queue)
            self.assertIsNotNone(task)
            assignments.append((robot_id, task.task_id))

        self.assertEqual(
            [task_id for _, task_id in assignments],
            list(range(10)),
        )
        self.assertEqual(
            [robot_id for robot_id, _ in assignments],
            idle_order,
        )
        self.assertEqual(len(queue), 0)

    def test_single_queued_task_cannot_be_assigned_twice(self):
        queue = TaskQueue()
        task = Task(7)
        queue.push(task)

        first_assignment = self.allocator.assign_next(self.robots[2], queue)
        second_assignment = self.allocator.assign_next(self.robots[4], queue)

        self.assertIs(first_assignment, task)
        self.assertIsNone(second_assignment)
        self.assertEqual(len(queue), 0)

    def test_queue_empties_exactly_once_across_many_assignment_requests(self):
        queue = TaskQueue()
        for task_id in range(10):
            queue.push(Task(task_id))

        assigned_ids = []
        request_order = [2, 0, 4, 1, 3] * 3

        for robot_id in request_order:
            task = self.allocator.assign_next(self.robots[robot_id], queue)
            if task is not None:
                assigned_ids.append(task.task_id)

        self.assertEqual(assigned_ids, list(range(10)))
        self.assertEqual(len(assigned_ids), len(set(assigned_ids)))
        self.assertEqual(len(queue), 0)

        # Once all ten tasks have been consumed, every later idle robot gets
        # None rather than a previously assigned task.
        for robot in self.robots.values():
            self.assertIsNone(self.allocator.assign_next(robot, queue))


if __name__ == "__main__":
    unittest.main()
