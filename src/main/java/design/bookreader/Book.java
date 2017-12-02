package design.bookreader;

/**
 * Created by tao on 10/24/17.
 */
public class Book {
    private int bookId;
    private String details;


    public Book(int bookId,String details){
        this.bookId = bookId;
        this.details = details;
    }
    public int getBookId() {
        return bookId;
    }

    public void setBookId(int bookId) {
        this.bookId = bookId;
    }

    public String getDetails() {
        return details;
    }

    public void setDetails(String details) {
        this.details = details;
    }

    public void update(){

    }
}
