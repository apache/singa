import singa.*;

public class Test {
  static {
    System.loadLibrary("singa_wrap");
  }

  public static void main(String argv[]) {
    Tensor t = new Tensor();
    System.out.println(t);
  }
}
