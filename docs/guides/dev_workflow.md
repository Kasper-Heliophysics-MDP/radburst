basic steps (add detail):

- create branch for feature/fix
    - Allows the user to test changes without affecting the main branch
- pull latest updates from main branch to avoid conflicts
    - Useful to ensure the user does not overwrite recent changes or do what someone else has already done
    - Also ensures that any code the user adds will be compatible with existing code
- make changes (update docs as needed)
- commit
    - These act as save points. If later code causes issues, a previous commit can be reverted to
- push
    - This allows others to see the branch the user has created and edit it
    - The branch stays seperate from the main branch
- pull request
    - Proposes merging the new branch with the main branch
    - Allows for changes to be considered by others; they may notice things the user who created the branch did not
- some team member reviews pr and gives feedback if they have any
    - example feedback: add comments for clarity, include corresponding documentation, improve variable naming, this part could be simplified
    - this process can keep team members informed,  encourage better code quality and help us learn from each other's work
- merge branch into main
    - Moves the changes made in the branch into the main branch
    - Ensures that other users will have the updated code when they make branches
- delete branch
    - The commits made are not deleted, only the branch itself

