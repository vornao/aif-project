:- dynamic maze/1.
:- dynamic start/2.


is_valid(Row, Col) :-
    maze(Maze),
    nth1(Row, Maze, RowList),   % get the row
    nth1(Col, RowList, 0).
    %Cell = 0.    % get the column


is_valid_action(Row, Col, Action) :-
    maze(Maze),
    action_to_coordinates(Action, Row, Col, NewRow, NewCol),
    is_valid(NewRow, NewCol).


valid_cardinal_actions(Row, Col, Action) :-
    maze(Maze),
    action_to_coordinates(Action, Row, Col, NewRow, NewCol),
    is_valid(NewRow, NewCol).



% translate action to coordinates where 0 means north, 1 means east, 2 means south, 3 means west
action_to_coordinates(0, Row, Col, NewRow, Col) :-  % north
    NewRow is Row - 1.                        % row - 1
action_to_coordinates(1, Row, Col, Row, NewCol) :-
    NewCol is Col + 1.
action_to_coordinates(2, Row, Col, NewRow, Col) :-
    NewRow is Row + 1.
action_to_coordinates(3, Row, Col, Row, NewCol) :-
    NewCol is Col - 1.

