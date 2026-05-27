#!/usr/bin/env python3
"""
Module for listing students sorted by average score.
"""


def top_students(mongo_collection):
    """
    Returns all students sorted by average score.

    Args:
        mongo_collection: pymongo collection object

    Returns:
        list of students with averageScore field
    """
    students = mongo_collection.aggregate([
        {
            "$addFields": {
                "averageScore": {
                    "$avg": "$topics.score"
                }
            }
        },
        {
            "$sort": {
                "averageScore": -1
            }
        }
    ])

    return list(students)
