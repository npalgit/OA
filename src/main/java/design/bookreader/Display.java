package design.bookreader;

/**
 * Created by tao on 10/24/17.
 */
public class Display {

    private Book activeBook;
    private User activeUser;
    private int pageNunber = 0;
    public void displayUser(User user){
        activeUser = user;
        refreshUsername();
    }

    public void displayBook(Book book){
        pageNunber = 0;
        activeBook = book;

        refreshTitle();
        refreshDetails();
        refreshPage();
    }

    public void refreshUsername(){

    }

    public void refreshTitle(){

    }

    public void refreshPage(){

    }

    public void refreshDetails(){

    }

    public void turnPageForward(){
        pageNunber++;
        refreshPage();
    }

    public void turnPageBackward(){
        pageNunber--;
        refreshPage();
    }
}
