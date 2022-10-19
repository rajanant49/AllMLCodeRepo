// SPDX-License-Identifier: GPL-3.0
//Group members: Ishan Goel(19CS30052) - Ashwamegh Rathore(19CS30009)
//Contract Address : 0x9327D39301A56b097131248371b369E3fB6e363e

pragma solidity ^0.8.7;

contract morra_game {

    uint gameStatus = 0; //0: no-init, 1: p1-init, 2:p2-init

    struct Player{
        int move;
        uint status; // 0: no-init, 1: init, 2: commited, 3: revealed
        address payable addr;
        uint betAmount;
        bytes32 moveCommit; // hash of <move><password1>
    }
    
    Player player1;
    Player player2;
    function initialize() public payable returns (uint){
        require(msg.value > 1e15 && gameStatus < 2, "0");
        if(gameStatus == 0){
            player1.addr = payable(msg.sender);
            player1.betAmount = msg.value;
            player1.status = 1;
            gameStatus = 1;
            return 1;
        }
        if(gameStatus == 1){
            require(player1.addr != msg.sender && msg.value >= player1.betAmount, "0");
            player2.addr = payable(msg.sender);
            player2.betAmount = msg.value;
            player2.status = 1;
            gameStatus = 2;
            return 2;
        }
        return 0;
    }

    function commitmove(bytes32 hashMove) public returns (bool){
        require(gameStatus == 2, "false");
        require(msg.sender == player1.addr || msg.sender == player2.addr, "false");
        if(msg.sender == player1.addr){
            require(player1.status == 1, "false");
            player1.moveCommit = hashMove;
            player1.status = 2;
        }
        if(msg.sender == player2.addr){
            require(player2.status == 1, "false");
            player2.moveCommit = hashMove;
            player2.status = 2;
        }
        return true;
    }

    function getFirstChar(string memory str) private pure returns (int) {
        if (bytes(str)[0] == 0x30) {
            return 0;
        } 
        else if (bytes(str)[0] == 0x31) {
            return 1;
        } 
        else if (bytes(str)[0] == 0x32) {
            return 2;
        } 
        else if (bytes(str)[0] == 0x33) {
            return 3;
        } 
        else if (bytes(str)[0] == 0x34) {
            return 4;
        } 
        else if (bytes(str)[0] == 0x35) {
            return 5;
        } 
        else {
            return -1;
        }
    }
    
    function revealmove(string memory revealedMove) public returns (int){
        int finMove = -1;
        require(msg.sender == player1.addr || msg.sender == player2.addr, "-1");
        require(player1.status >= 2 && player2.status >= 2, "-1");
        if(msg.sender == player1.addr){
            require(sha256(abi.encodePacked(revealedMove)) == player1.moveCommit, "-1");
            player1.move = getFirstChar(revealedMove);
            require(player1.move != -1, "-1");
            player1.status = 3;
            finMove = player1.move;
        }
        if(msg.sender == player2.addr){
            require(sha256(abi.encodePacked(revealedMove)) == player2.moveCommit, "-1");
            player2.move = getFirstChar(revealedMove);
            require(player2.move != -1, "-1");
            player2.status = 3;
            finMove = player2.move;
        }
        if(player1.status == 3 && player2.status == 3){
            if(player1.move == player2.move){
                player2.addr.transfer(address(this).balance);
            }else{
                player1.addr.transfer(address(this).balance);
            }
            gameStatus = 0;
            player1.status = 0;
            player2.status = 0;
        }
        return finMove;
    }

    function getBalance() public view returns (uint){
        return address(this).balance;
    }

    function getPlayerId() public view returns (uint){
        if(msg.sender == player1.addr){
            return 1;
        }
        if(msg.sender == player2.addr){
            return 2;
        }
        return 0;
    }
}
