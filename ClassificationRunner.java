import edu.stanford.nlp.classify.*;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.Iterator;
import org.apache.poi.ss.usermodel.Cell;  
import org.apache.poi.ss.usermodel.Row;  
import org.apache.poi.xssf.usermodel.XSSFSheet;  
import org.apache.poi.xssf.usermodel.XSSFWorkbook;  

public class ClassificationRunner
{
    public static final int DATA_COL = 5;
    public static final int LABEL_COL = 6;

    public static void main(String [] args)
    {


        try {
            File file = new File("SFUcorpus.xlsx");
            FileInputStream fis = new FileInputStream(file);

            XSSFWorkbook web = new XSSFWorkbook(fis);
            XSSFSheet sheet = wb.getSheetAt(0);
            Iterator<Row> itr = sheet.iterator();
            String data, label;

            while(itr.hasNext())
            {
                Row row = itr.next();
                data = row.getCell(DATA_COL);
                label = row.getCell(LABEL_COL);
            }

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        LogisticClassifierFactory<int[], String> factory = new LogisticClassifierFactory();


        LogisticClassifier<int[], String> logClass = factory.trainClassifier(data);
    }
}